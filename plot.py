import pandas as pd
import matplotlib.pyplot as plt
import argparse

def plot(model_path):

    target1 = f'work_dir/{model_path}/result.csv'
    target2 = f'work_dir/{model_path}/pre_result.csv'
    targets = [target1, target2]

    for target in targets:
        data = pd.read_csv(target)

        data = data.sort_values(by='frame')

        # Extract data
        frames = data['frame']
        loss = data['loss']
        accuracy = data['accuracy']

        # Setup high resolution plotting
        fig, ax1 = plt.subplots(figsize=(10, 6), dpi=120)

        # Plot loss
        color = 'tab:red'
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Loss', color=color)
        ax1.plot(frames, loss, color=color, label='Loss')
        ax1.tick_params(axis='y', labelcolor=color)

        # Plot accuracy
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Accuracy', color=color)
        ax2.plot(frames, accuracy, color=color, label='Accuracy')
        ax2.tick_params(axis='y', labelcolor=color)

        # Title and layout adjustments
        plt.title('Loss and Accuracy over Frames')
        fig.tight_layout()

        # Show plot
        # plt.show()
        if target == target1:
            plot_name = "training_plot.png"
        elif target == target2:
            plot_name = "prediction_plot.png"

        plt.savefig(f"work_dir/{model_path}/{plot_name}")
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train model with different configurations.")
    parser.add_argument("--model_path", type=str, required=True, help="model path")
    args = parser.parse_args()
    model_path = args.model_path

    plot(model_path)