import matplotlib.pyplot as plt
import numpy as np

def plot_single_iteration_returns(returns):
    fig, axes = plt.subplots(1)
    fig.set_figheight(5)
    fig.set_figwidth(25)

    axes.plot(returns)

    axes.set_xlabel("Episodes")
    axes.set_ylabel("Discounted Reward Achieved")
    axes.legend()

    plt.show
    

def plot_returns(
    all_returns,
    labels,
):
    fig, axes = plt.subplots(1)
    fig.set_figheight(5)
    fig.set_figwidth(25)

    colors = ["blue", "red", "green", "orange", "purple"]

    for i, returns in enumerate(all_returns):
        # mean is just the average of all the returns at each episode for this agent
        mean = np.average(returns, axis=0)

        axes.plot(mean, label=labels[i], color=colors[i])

        std = np.std(returns, axis=0)

        std_error = std * (1 / np.sqrt(len(returns)))

        # axes.fill_between(
        #     np.arange(num_episodes),
        #     mean - (1.96 * std_error),
        #     mean + (1.96 * std_error),
        #     alpha=0.2,
        #     color=colors[i],
        # )


    # axes.axhline(
    #     y=calculate_max_return(),
    #     label="Max Possible Return",
    #     color="black",
    #     linestyle="dashed",
    # )

    axes.set_xlabel("Episodes")
    axes.set_ylabel("Discounted Reward Achieved")
    axes.legend()

    plt.show


