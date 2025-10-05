import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from torch.multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Data directories
    data_dir = 'new_database'
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    # Data transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Load datasets
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, data_transforms['train']),
        'val': datasets.ImageFolder(val_dir, data_transforms['val']),
        'test': datasets.ImageFolder(test_dir, data_transforms['test']),
    }

    # Data loaders
    batch_size = 32
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=0),
        'test': DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False, num_workers=0),
    }

    # Class names
    class_names = image_datasets['train'].classes
    print(f"Classes: {class_names}")

    # Calculate class weights for imbalance
    train_labels = [label for _, label in image_datasets['train']]
    class_counts = np.bincount(train_labels)
    class_weights = 1. / class_counts
    class_weights = torch.FloatTensor(class_weights).cuda() if torch.cuda.is_available() else torch.FloatTensor(class_weights)

    # Model
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training function
    def train_model(model, criterion, optimizer, num_epochs=25):
        import time
        import torch.cuda.profiler as profiler
        import torch.autograd.profiler as autograd_profiler

        best_model_wts = model.state_dict()
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            start_epoch_time = time.time()

            for phase in ['train', 'val']:
                print(f'Starting {phase} phase')
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                batch_count = 0

                for inputs, labels in dataloaders[phase]:
                    print(f'Loading batch {batch_count + 1} in {phase} phase')
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        with autograd_profiler.profile(use_cuda=torch.cuda.is_available()) as prof:
                            outputs = model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)

                            if phase == 'train':
                                loss.backward()
                                optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    batch_count += 1
                    if batch_count % 10 == 0:
                        print(f'{phase} Batch {batch_count} processed')
                        print(prof.key_averages().table(sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total", row_limit=5))

                    # Save checkpoint every 50 batches during training
                    if phase == 'train' and batch_count % 50 == 0:
                        checkpoint_path = f'tb_detector_checkpoint_epoch{epoch}_batch{batch_count}.pth'
                        torch.save(model.state_dict(), checkpoint_path)
                        print(f'Saved checkpoint: {checkpoint_path}')

                epoch_loss = running_loss / len(image_datasets[phase])
                epoch_acc = running_corrects.double() / len(image_datasets[phase])

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict()

            epoch_duration = time.time() - start_epoch_time
            print(f'Epoch {epoch} completed in {epoch_duration:.2f} seconds\n')

        print(f'Best val Acc: {best_acc:.4f}')
        model.load_state_dict(best_model_wts)
        return model

    # Train the model
    model = train_model(model, criterion, optimizer, num_epochs=10)

    # Save the model
    torch.save(model.state_dict(), 'tb_detector_resnet50.pth')

    # Evaluate on test set
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("Test Results:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
