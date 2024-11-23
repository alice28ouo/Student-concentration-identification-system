import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def load_known_faces(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def analyze_face_data(known_faces):
    name_to_descriptors = defaultdict(list)
    for face in known_faces:
        name_to_descriptors[face['name']].extend(face['descriptors'])

    accuracies = {}
    for name, descriptors in name_to_descriptors.items():
        descriptors = np.array(descriptors)
        distances = np.array([[euclidean_distance(d1, d2) for d2 in descriptors] for d1 in descriptors])
        np.fill_diagonal(distances, np.inf)  # Ignore self-distance
        min_distances = distances.min(axis=1)
        accuracy = (min_distances < 0.6).mean()  # 0.6 is a threshold, can be adjusted as needed
        accuracies[name] = accuracy

    return accuracies, name_to_descriptors

def plot_accuracies(accuracies):
    names = list(accuracies.keys())
    acc_values = list(accuracies.values())

    plt.figure(figsize=(10, 6))
    plt.bar(names, acc_values)
    plt.title('Face Recognition Accuracy')
    plt.xlabel('Name')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    for i, v in enumerate(acc_values):
        plt.text(i, v, f'{v:.2f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig('face_recognition_accuracy.png')
    plt.close()

def plot_descriptor_distribution(name_to_descriptors):
    plt.figure(figsize=(12, 6))
    for name, descriptors in name_to_descriptors.items():
        descriptors = np.array(descriptors)
        mean_descriptor = descriptors.mean(axis=0)
        plt.plot(mean_descriptor, label=name)
    
    plt.title('Face Descriptor Distribution')
    plt.xlabel('Feature Dimension')
    plt.ylabel('Feature Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig('face_descriptor_distribution.png')
    plt.close()

def plot_individual_distributions(name_to_descriptors):
    for name, descriptors in name_to_descriptors.items():
        descriptors = np.array(descriptors)
        mean_descriptor = descriptors.mean(axis=0)
        std_descriptor = descriptors.std(axis=0)

        plt.figure(figsize=(12, 6))
        plt.plot(mean_descriptor, label='Mean')
        plt.fill_between(range(len(mean_descriptor)), 
                         mean_descriptor - std_descriptor, 
                         mean_descriptor + std_descriptor, 
                         alpha=0.2, label='Standard Deviation')
        plt.title(f'Face Descriptor Distribution for {name}')
        plt.xlabel('Feature Dimension')
        plt.ylabel('Feature Value')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'face_descriptor_distribution_{name}.png')
        plt.close()

def main():
    filename = 'merged_6faces.pkl'
    known_faces = load_known_faces(filename)
    
    accuracies, name_to_descriptors = analyze_face_data(known_faces)
    plot_accuracies(accuracies)
    plot_descriptor_distribution(name_to_descriptors)
    plot_individual_distributions(name_to_descriptors)

    print("Analysis complete. Charts have been saved.")
    print("\nFace recognition accuracy for each person:")
    for name, accuracy in accuracies.items():
        print(f"{name}: {accuracy:.2f}")

if __name__ == "__main__":
    main()