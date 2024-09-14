from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
def compute_f1(predictions: list, labels: list, label_names: list) -> float:
    """
    Computes the F1 score for the given predictions and labels.

    Args:
        predictions (list): List of predicted labels.
        labels (list): List of true labels.
        label_names (list): List of all possible label names.

    Returns:
        float: Macro-averaged F1 score.
    """
    clean_labels = []
    clean_predictions = []

    for i in range(len(labels)):
        named_labels = []
        named_preds = []
        for j in range(len(labels[i])):
            if labels[i][j] != -100:
                named_labels.append(label_names[labels[i][j]])
                named_preds.append(label_names[predictions[i][j]])

        clean_labels.append(named_labels)
        clean_predictions.append(named_preds)

    mlb = MultiLabelBinarizer(classes=label_names)
    binarized_labels = mlb.fit_transform(clean_labels)
    binarized_predictions = mlb.transform(clean_predictions)

    return f1_score(binarized_labels, binarized_predictions, average="macro")

