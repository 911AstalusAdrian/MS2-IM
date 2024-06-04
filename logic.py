import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier


from encoders import OneHotEncoder, KMerEncoder
from data_processing import load_and_preprocess_data, process_sequences
from algorithms import CNNClassifier



# Evaluate the models result regarding the input data
def evaluate_algorithm(model, model_type, encoder, sequence, max_length):
    user_input_sequence = process_sequences(sequence, max_length)
    encoded_user_input = encoder.encode(user_input_sequence, max_length, 4)
    if model_type == "cnn":
        encoded_user_input = encoded_user_input.unsqueeze(0)  # Add batch dimension
        cnn_output = model(encoded_user_input.permute(0, 2, 1))
        probabilities = F.softmax(cnn_output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        print(f'Predicted Class: {"Modern" if predicted_class == 1 else "Ancient"}')
        return predicted_class == 1
    elif model_type == "rf":
        encoded_user_input_reshaped = encoded_user_input.reshape(1, -1)  # Make it 2D array
        prediction = model.predict(encoded_user_input_reshaped)
        return prediction == 1


def train_cnn_k_fold(model, data_x, data_y, num_epochs=5, batch_size=32, n_splits=5):
    kfold = KFold(n_splits=n_splits, shuffle=True)
    results = {}

    for fold, (train_ids, val_ids) in enumerate(kfold.split(data_x)):
        print(f"FOLD {fold}")
        print("--------------------------------")

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

        trainloader = DataLoader(TensorDataset(data_x, data_y),
                                 batch_size=batch_size, sampler=train_subsampler)
        valloader = DataLoader(TensorDataset(data_x, data_y),
                               batch_size=batch_size, sampler=val_subsampler)

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(0, num_epochs):
            print(f'Starting epoch {epoch + 1}')
            current_loss = 0.0

            for i, (inputs, labels) in enumerate(trainloader, 0):
                optimizer.zero_grad()
                outputs = model(inputs.permute(0, 2, 1))
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                current_loss += loss.item()
                if i % 10 == 9:
                    print(f'Loss after mini-batch {i + 1}: {current_loss / 10:.3f}')
                    current_loss = 0.0

        correct, total = 0, 0
        y_true = []
        y_pred = []

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(valloader, 0):
                outputs = model(inputs.permute(0, 2, 1))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                y_true.extend(labels.tolist())
                y_pred.extend(predicted.tolist())

            acc = 100.0 * (correct / total)
            f1 = f1_score(y_true, y_pred, average='macro')
            recall = recall_score(y_true, y_pred, average='macro')

            print(f'Accuracy for fold {fold}: {acc}%')
            print(f'F1 Score for fold {fold}: {f1}')
            print(f'Recall for fold {fold}: {recall}')

            results[fold] = {'Accuracy': acc, 'F1 Score': f1, 'Recall': recall}

    print(f"K-FOLD CROSS VALIDATION RESULTS FOR {n_splits} FOLDS")
    print("================================")
    avg_acc = 0.0
    avg_f1 = 0.0
    avg_recall = 0.0

    for key, value in results.items():
        print(f'Fold {key}: Accuracy {value["Accuracy"]}% - F1 Score {value["F1 Score"]} - Recall {value["Recall"]}')
        avg_acc += value['Accuracy']
        avg_f1 += value['F1 Score']
        avg_recall += value['Recall']


    fin_acc = avg_acc / len(results.items())
    fin_f1 = avg_f1 / len(results.items())
    fin_recall = avg_recall / len(results.items())

    print(f'Average: Accuracy {avg_acc / len(results.items())}% - F1 Score {avg_f1 / len(results.items())} - Recall {avg_recall / len(results.items())}')
    return fin_acc, fin_f1, fin_recall



def train_rf_k_fold(data_x, data_y, n_estimators=100, n_splits=5):
    kfold = KFold(n_splits=n_splits, shuffle=True)
    results = {}

    for fold, (train_ids, val_ids) in enumerate(kfold.split(data_x)):
        print(f"FOLD {fold}")
        print("--------------------------------")

        train_x, train_y = data_x[train_ids], data_y[train_ids]
        val_x, val_y = data_x[val_ids], data_y[val_ids]

        model = RandomForestClassifier(n_estimators=n_estimators)

        # Reshape data if needed
        train_x_reshaped = train_x.reshape(train_x.shape[0], -1)
        val_x_reshaped = val_x.reshape(val_x.shape[0], -1)

        # Train the model
        model.fit(train_x_reshaped, train_y)

        # Validate the model
        predictions = model.predict(val_x_reshaped)

        # Compute metrics
        acc = 100.0 * accuracy_score(val_y, predictions)
        f1 = f1_score(val_y, predictions, average='macro')
        recall = recall_score(val_y, predictions, average='macro')

        print(f'Accuracy for fold {fold}: {acc}%')
        print(f'F1 Score for fold {fold}: {f1}')
        print(f'Recall for fold {fold}: {recall}')

        results[fold] = {'Accuracy': acc, 'F1 Score': f1, 'Recall': recall}

    print(f"K-FOLD CROSS VALIDATION RESULTS FOR {n_splits} FOLDS")
    print("================================")
    avg_acc = 0.0
    avg_f1 = 0.0
    avg_recall = 0.0

    for key, value in results.items():
        print(f'Fold {key}: Accuracy {value["Accuracy"]}% - F1 Score {value["F1 Score"]} - Recall {value["Recall"]}')
        avg_acc += value['Accuracy']
        avg_f1 += value['F1 Score']
        avg_recall += value['Recall']


    fin_acc = avg_acc/len(results.items())
    fin_f1 = avg_f1/len(results.items())
    fin_recall = avg_recall/len(results.items())

    print(f'Average: Accuracy {avg_acc/len(results.items())}% - F1 Score {avg_f1/len(results.items())} - Recall {avg_recall/len(results.items())}')

    return model, fin_acc, fin_f1, fin_recall


def run_main_logic(user_input_sequence, encoder_choice, algorithm_choice):
    if encoder_choice == "OneHotEncoder":
        encoder = OneHotEncoder()
        input_channels = 4
    elif encoder_choice == "KMerEncoder":
        encoder = KMerEncoder()
        input_channels = len(encoder.kmer_to_idx)

    # Load and process data
    data_x, data_y, max_length = load_and_preprocess_data(encoder)

    # If KMerEncoder is chosen, adjust the max_length for sequence processing
    if isinstance(encoder, KMerEncoder):
        max_length = max_length - encoder.k + 1


    if algorithm_choice == "CNN Classifier":
        # Instantiate the CNN model
        input_channels = 4
        num_classes = 2  # Ancient vs. Modern
        embedding_dim = None
        model = CNNClassifier(input_channels, num_classes, embedding_dim)
        result = evaluate_algorithm(model, "cnn", encoder, user_input_sequence, max_length)

        accuracy, f1, recall = train_cnn_k_fold(model, data_x, data_y)

    elif algorithm_choice == "Random Forest Algorithm":
        model, accuracy, f1, recall = train_rf_k_fold(data_x, data_y)
        result = evaluate_algorithm(model, "rf", encoder, user_input_sequence, max_length)

    return result, accuracy, f1, recall

