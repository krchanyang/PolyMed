import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

def basic_SMOTE(data_x, data_y):
    train_x = np.array(data_x)
    train_y = np.array(data_y)

    print("Before oversampling: ", Counter(train_y))

    # Find the number of samples needed for each class to match the class with the maximum count
    max_samples = max(Counter(train_y).values())

    # SMOTE Application
    smote = SMOTE(sampling_strategy='auto', k_neighbors=2)

    can_smote = [label for label, count in Counter(train_y).items() if count > 1]
    cannot_smote = [label for label, count in Counter(train_y).items() if count == 1]

    X_res, y_res = train_x, train_y

    for label in can_smote:
        smote.sampling_strategy = {label: max_samples}
        X_res, y_res = smote.fit_resample(X_res, y_res)

    # Handling the cannot_smote classes with random symptom sampling
    for label in cannot_smote:
        instance = train_x[train_y == label][0]
        samples = random_symptom_sampling(instance, max_samples)
        X_res = np.vstack((X_res, samples))
        y_res = np.concatenate((y_res, [label] * max_samples))

    print("After oversampling: ", Counter(y_res))

    # Distribution before SMOTE
    pre_smote_counts = Counter(train_y)

    # Distribution after SMOTE
    post_smote_counts = Counter(y_res)

    # Plot
    plt.figure(figsize=(15, 6))

    # Before SMOTE
    plt.subplot(1, 2, 1)
    sns.barplot(x=list(pre_smote_counts.keys()), y=list(pre_smote_counts.values()), color='skyblue')
    plt.title('Label Distribution Before SMOTE')
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.xticks(rotation=45)

    # After SMOTE
    plt.subplot(1, 2, 2)
    sns.barplot(x=list(post_smote_counts.keys()), y=list(post_smote_counts.values()), color='salmon')
    plt.title('Label Distribution After SMOTE')
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

    return X_res, y_res


def balance_SMOTE(data_x, data_y, amplification=4):
    train_x = np.array(data_x)
    train_y = np.array(data_y)

    print("Before oversampling: ", Counter(train_y))

    # Determine the "balance point" using the median class size.
    can_smote_range = np.where(np.bincount(train_y.astype(int)) > 1)[0]
    balance_point = int(np.mean(list(Counter(train_y[can_smote_range]).values()))) * amplification

    # Set up our oversampling and undersampling methods
    smote = SMOTE(sampling_strategy='auto', k_neighbors=2)
    under_sampler = RandomUnderSampler(sampling_strategy={cls: balance_point for cls, count in Counter(train_y).items() if count > balance_point})

    # First, oversample to the balance point
    can_smote = [label for label, count in Counter(train_y).items() if count < balance_point and count > 1]
    cannot_smote = [label for label, count in Counter(train_y).items() if count == 1]

    X_res, y_res = train_x, train_y
    for label in can_smote:
        smote.sampling_strategy = {label: balance_point}
        X_res, y_res = smote.fit_resample(X_res, y_res)

    # Now, apply the random symptom sampling for cannot_smote classes
    for label in cannot_smote:
        single_instance = train_x[train_y == label]
        new_samples = random_symptom_sampling(single_instance[0], balance_point)
        X_res = np.vstack((X_res, new_samples))
        y_res = np.concatenate((y_res, [label] * balance_point))

    # Then, undersample classes above the balance point
    X_res, y_res = under_sampler.fit_resample(X_res, y_res)

    print("After oversampling: ", Counter(y_res))

    # Distribution before SMOTE
    pre_smote_counts = Counter(train_y)

    # Distribution after SMOTE
    post_smote_counts = Counter(y_res)

    # Plot
    plt.figure(figsize=(15, 6))

    # Before SMOTE
    plt.subplot(1, 2, 1)
    sns.barplot(x=list(pre_smote_counts.keys()), y=list(pre_smote_counts.values()), color='skyblue')
    plt.title('Label Distribution Before SMOTE')
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.xticks(rotation=45)

    # After SMOTE
    plt.subplot(1, 2, 2)
    sns.barplot(x=list(post_smote_counts.keys()), y=list(post_smote_counts.values()), color='salmon')
    plt.title('Label Distribution After SMOTE')
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

    return X_res, y_res


def random_symptom_sampling(instance, target_count):
    indices = np.where(instance == 1)[0]
    new_instances = []
    if len(indices) == 0:  # If there are no positive features, replicate the instance
        return np.tile(instance, (target_count, 1))
    for _ in range(target_count - 1):
        np.random.shuffle(indices)
        n_selected_features = np.random.randint(1, len(indices) + 1)
        new_instance = np.zeros_like(instance)
        new_instance[indices[:n_selected_features]] = 1
        new_instances.append(new_instance)
    return np.vstack([instance] + new_instances)


def basic_SMOTE_Tomek(data_x, data_y):
    train_x = np.array(data_x)
    train_y = np.array(data_y)

    print("Before resampling: ", Counter(train_y))

    # Identify classes with only one instance
    class_counts = Counter(train_y)
    single_instance_classes = [cls for cls, count in class_counts.items() if count == 1]

    # Separate single instance classes from the rest of the dataset
    mask_single_instance = np.isin(train_y, single_instance_classes)
    X_single_instance = train_x[mask_single_instance]
    y_single_instance = train_y[mask_single_instance]

    X_multi_instance = train_x[~mask_single_instance]
    y_multi_instance = train_y[~mask_single_instance]

    # Set up SMOTE-Tomek for combined over-sampling and cleaning
    smote_tomek = SMOTETomek(sampling_strategy='auto')

    # Resample the dataset using SMOTE-Tomek
    X_res, y_res = smote_tomek.fit_resample(X_multi_instance, y_multi_instance)

    # Determine the target count for oversampling
    target_count = max(Counter(y_res).values())

    # Oversample single-instance classes using random symptom sampling
    for instance, label in zip(X_single_instance, y_single_instance):
        oversampled_instances = random_symptom_sampling(instance, target_count)
        X_res = np.vstack((X_res, oversampled_instances))
        y_res = np.concatenate((y_res, [label] * target_count))

    print("After resampling: ", Counter(y_res))

    # Distribution before SMOTE-Tomek
    pre_resample_counts = Counter(train_y)

    # Distribution after SMOTE-Tomek
    post_resample_counts = Counter(y_res)

    # Plot
    plt.figure(figsize=(15, 6))

    # Before SMOTE-Tomek
    plt.subplot(1, 2, 1)
    sns.barplot(x=list(pre_resample_counts.keys()), y=list(pre_resample_counts.values()), color='skyblue')
    plt.title('Label Distribution Before SMOTE-Tomek')
    plt.ylabel('Count')
    plt.xlabel('Labels')
    plt.xticks([], [])  # Removing tick labels but keeping x-axis label

    # After SMOTE-Tomek
    plt.subplot(1, 2, 2)
    keys = list(post_resample_counts.keys())
    original_counts = [pre_resample_counts.get(key, 0) for key in keys]
    additional_counts = [post_resample_counts[key] - original_count for key, original_count in zip(keys, original_counts)]
    sns.barplot(x=keys, y=original_counts, color='skyblue')
    sns.barplot(x=keys, y=additional_counts, bottom=original_counts, color='salmon')
    plt.title('Label Distribution After SMOTE-Tomek')
    plt.ylabel('Count')
    plt.xlabel('Labels')
    plt.xticks([], [])  # Removing tick labels but keeping x-axis label

    plt.tight_layout()
    plt.show()

    return X_res, y_res


def group_by_label(data_x, data_y):
    grouped = defaultdict(list)
    for x, y in zip(data_x, data_y):
        grouped[y].append(x)
    return grouped


def random_symptom_sampling_ea(instance):
    indices = np.where(instance > 0)[0]
    if len(indices) == 0:
        return instance  # Replicate the instance if there are no positive features
    n_selected_features = np.random.randint(1, len(indices) + 1)
    np.random.shuffle(indices)
    new_instance = np.zeros_like(instance)
    new_instance[indices[:n_selected_features]] = 1
    return new_instance


def finetune_sampling(data_x, data_y, target_count_per_label):
    grouped_data = group_by_label(data_x, data_y)
    sampled_data_x = []
    sampled_data_y = []

    for label, instances in grouped_data.items():
        if len(instances) == 1:
            merged_instance = instances[0]
        else:
            merged_instance = np.sum(instances, axis=0)

        for _ in range(target_count_per_label):
            sampled_instance = random_symptom_sampling_ea(merged_instance)
            sampled_data_x.append(sampled_instance)
            sampled_data_y.append(label)

    return np.array(sampled_data_x), np.array(sampled_data_y)