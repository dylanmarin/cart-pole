import matplotlib.pyplot as plt
import numpy as np

def plot_single_iteration_returns(returns, y_label="Discounted Reward Achieved"):
    fig, axes = plt.subplots(1)
    fig.set_figheight(5)
    fig.set_figwidth(25)

    axes.plot(returns)

    axes.set_xlabel("Episodes")
    axes.set_ylabel(y_label)
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



def plot_returns_different_agents(
    all_returns,
    labels,
):
    fig, axes = plt.subplots(1)
    fig.set_figheight(5)
    fig.set_figwidth(25)

    colors = ["blue", "red", "green", "orange", "purple"]

    for i, returns in enumerate(all_returns):
        axes.plot(returns, label=labels[i], color=colors[i])

    axes.set_xlabel("Episodes")
    axes.set_ylabel("Discounted Reward Achieved")
    axes.legend()

    plt.show



def plot_episode_lengths(
    all_episode_lengths, # array of arrays (one for each agent type) of arrays of episode lengths
    labels,
    smoothing = 1,
    error_range=True,
):
    fig, axes = plt.subplots(1)
    fig.set_figheight(5)
    fig.set_figwidth(25)

    colors = ["blue", "red", "green", "orange", "purple", "black", "pink"]

    for i, episode_lengths in enumerate(all_episode_lengths):
        label = labels[i]

        # mean is just the average of all the returns at each episode for this agent
        mean = np.mean(episode_lengths, axis=0)

        if smoothing > 1:
            # plot the average of every smoothing points (e.g. if smoothing = 10, plot the average of every 10 points)
            mean = np.convolve(mean, np.ones(smoothing), "valid") / smoothing

        axes.plot(mean, label=labels[i], color=colors[i])

        if error_range:
            std = np.std(episode_lengths, axis=0)
            std_error = std * (1 / np.sqrt(len(episode_lengths)))
        
            if smoothing > 1:
                std_error = np.convolve(std_error, np.ones(smoothing), "valid") / smoothing

            axes.fill_between(
                np.arange(len(mean)),
                mean - (1.96 * std_error),
                mean + (1.96 * std_error),
                alpha=0.2,
                color=colors[i],
            )

    # axes.axhline(
    #     y=calculate_max_return(),
    #     label="Max Possible Return",
    #     color="black",
    #     linestyle="dashed",
    # )

    axes.set_xlabel("Episodes")
    axes.set_ylabel("Avg Episode Length")
    axes.legend()

    plt.show
