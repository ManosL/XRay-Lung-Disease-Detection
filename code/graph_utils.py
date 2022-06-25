from plotly.subplots import make_subplots
import plotly.graph_objects as go


def plot_train_val(epochs, train_losses, val_losses, network_name = 'our model'):
    epochs_stopped = len(train_losses)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=list(range(epochs_stopped)), y=train_losses,
                        mode='lines', name='Train Loss'))

    fig.add_trace(go.Scatter(x=list(range(epochs_stopped)), y=val_losses,
                        mode='lines', name='Validation Loss'))

    title = 'Train and Validation Loss after training ' + network_name + ' for ' + str(epochs) + 'epochs'
    title = title + '(stopped at ' + str(epochs_stopped) + ' epochs).'

    fig.update_layout(title=title, xaxis_title='Epochs', yaxis_title='Cross-Entropy Loss')

    fig.show()

    return




def plot_confusion_matrix(conf_matrix, confusion_matrix_labels):
    fig = go.Figure()

    fig.add_trace(go.Heatmap(z=conf_matrix, text=conf_matrix, x=confusion_matrix_labels,
                            y=confusion_matrix_labels, texttemplate='%{text}', textfont={"size":20},
                            legendgroup=1, showlegend=True))

    fig.update_annotations(font_size=14)

    fig.show()
    return