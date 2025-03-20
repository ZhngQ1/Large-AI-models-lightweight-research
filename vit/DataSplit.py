import torch
import torch.utils.data as data
import numpy as np
    
    
def data_noniid(dataset, num_users):
    num_clients = num_users
    num_classes = len(dataset.classes)
    classes_per_client = num_classes // num_clients

    client_datasets = []
    for i in range(num_clients):
        start_class = i * classes_per_client
        end_class = (i + 1) * classes_per_client
        client_classes = dataset.classes[start_class:end_class]
        
        # 选择对应类别的样本
        client_indices = [idx for idx, (_, label) in enumerate(dataset.samples) if label in client_classes]
        client_data = data.Subset(dataset, client_indices)
        client_datasets.append(client_data)

    data_loader = []
    for i in range(num_clients):
        data_loader.append(data.DataLoader(
            client_datasets[i], 
            batch_size=64,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        ))

    return data_loader


def data_noniid_with_public(dataset, num_users):   
    num_clients = num_users
    num_data = len(dataset)
    num_classes = len(dataset.classes)
    classes_per_client = num_classes // num_clients

    # Calculate the number of public data samples
    num_public_data = int(0.2 * num_data)
    num_private_data = num_data - num_public_data

    # Shuffle the dataset indices
    indices = torch.randperm(num_data)

    # Split the indices for public and private datasets
    public_indices = indices[:num_public_data]
    private_indices = indices[num_public_data:]

    # Create the public dataset
    public_dataset = data.Subset(dataset, public_indices)

    # Create a data loader for the public dataset
    public_data_loader = data.DataLoader(
        public_dataset,
        batch_size=64,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    client_datasets = []
    for i in range(num_clients):
        start_class = i * classes_per_client
        end_class = (i + 1) * classes_per_client
        client_classes = dataset.classes[start_class:end_class]
        
        # 选择对应类别的样本
        client_indices = [idx for idx, (_, label) in enumerate(dataset.samples) 
                        if idx in private_indices and label in client_classes]
        client_data = data.Subset(dataset, client_indices)
        client_datasets.append(client_data)

    data_loader = []
    for i in range(num_clients):
        data_loader.append(data.DataLoader(
            client_datasets[i], 
            batch_size=64,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        ))
    
    return data_loader, public_data_loader


def data_dirichlet(dataset, num_users):
    num_clients = num_users
    num_classes = len(dataset.classes)
    
    client_datasets = [[] for _ in range(num_clients)]
    
    # Get labels for all data
    all_labels = [label for _, label in dataset.samples]

    # Dirichlet distribution parameter
    alpha = 0.5

    # Generate Dirichlet distribution for each class
    for c in range(num_classes):
        class_indices = [idx for idx, label in enumerate(all_labels) if label == c]
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        client_indices = np.split(class_indices, proportions)
        
        for i in range(num_clients):
            client_datasets[i].extend(client_indices[i])

    data_loader = []
    for i in range(num_clients):
        client_data = data.Subset(dataset, client_datasets[i])
        data_loader.append(data.DataLoader(
            client_data, 
            batch_size=64,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        ))

    return data_loader


def data_dirichlet_with_public(dataset, num_users):
    num_clients = num_users
    num_data = len(dataset)
    num_classes = len(dataset.classes)
    
    # Calculate the number of public data samples
    num_public_data = int(0.2 * num_data)
    num_private_data = num_data - num_public_data

    # Shuffle the dataset indices
    indices = torch.randperm(num_data)

    # Split the indices for public and private datasets
    public_indices = indices[:num_public_data]
    private_indices = indices[num_public_data:]

    # Create the public dataset
    public_dataset = data.Subset(dataset, public_indices)

    # Create a data loader for the public dataset
    public_data_loader = data.DataLoader(
        public_dataset,
        batch_size=64,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    # Dirichlet distribution parameter
    alpha = 0.1
    client_datasets = [[] for _ in range(num_clients)]
    
    # Get labels for private data
    private_labels = [dataset.samples[idx][1] for idx in private_indices]

    # Generate Dirichlet distribution for each class
    for c in range(num_classes):
        class_indices = [idx for idx, label in zip(private_indices, private_labels) if label == c]
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        client_indices = np.split(class_indices, proportions)
        
        for i in range(num_clients):
            client_datasets[i].extend(client_indices[i])

    data_loader = []
    for i in range(num_clients):
        client_data = data.Subset(dataset, client_datasets[i])
        data_loader.append(data.DataLoader(
            client_data, 
            batch_size=64,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        ))
    
    return data_loader, public_data_loader


def data_iid(dataset, num_users):
    num_clients = num_users
    num_data = len(dataset)
    data_per_client = num_data // num_clients

    indices = torch.randperm(len(dataset))
    shuffled_dataset = data.Subset(dataset, indices)

    client_datasets = []
    for i in range(num_clients):
        start = i * data_per_client
        end = (i + 1) * data_per_client
        client_data = data.Subset(shuffled_dataset, range(start, end))
        client_datasets.append(client_data)

    data_loader = []
    for i in range(num_clients):
        data_loader.append(data.DataLoader(
            client_datasets[i], 
            batch_size=64,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        ))

    return data_loader


def data_iid_with_public(dataset, num_users):
    num_clients = num_users
    num_data = len(dataset)

    # Calculate the number of public data samples
    num_public_data = int(0.2 * num_data)
    num_private_data = num_data - num_public_data

    # Shuffle the dataset indices
    indices = torch.randperm(num_data)
    
    # Split the indices for public and private datasets
    public_indices = indices[:num_public_data]
    private_indices = indices[num_public_data:]

    # Create the public dataset
    public_dataset = data.Subset(dataset, public_indices)

    # Create a data loader for the public dataset
    public_data_loader = data.DataLoader(
        public_dataset,
        batch_size=64,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    data_per_client = num_private_data // num_clients

    shuffled_private_dataset = data.Subset(dataset, private_indices)

    client_datasets = []
    for i in range(num_clients):
        start = i * data_per_client
        end = (i + 1) * data_per_client
        client_data = data.Subset(shuffled_private_dataset, range(start, end))
        client_datasets.append(client_data)

    data_loaders = []
    for i in range(num_clients):
        data_loaders.append(data.DataLoader(
            client_datasets[i], 
            batch_size=64,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        ))

    return data_loaders, public_data_loader

