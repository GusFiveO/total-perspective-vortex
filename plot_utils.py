from matplotlib import pyplot as plt


def plot_signals_with_events(
    signals,
    channel_names,
    times,
    freq,
    events,
    event_id,
    title,
    output_path,
):
    n_channels = len(signals)
    fig, axes = plt.subplots(
        n_channels, 1, figsize=(14, 2 * n_channels), sharex=True
    )
    fig.suptitle(title, fontsize=16)

    for i, ax in enumerate(axes):
        ax.plot(times, signals[i], color="black", linewidth=0.8)
        ax.set_ylabel(channel_names[i], rotation=0, labelpad=40)
        # ax.tick_params(left=False, labelleft=False)

        # Add vertical lines for events
        for t, _, label in events:
            t = t / freq
            ax.axvline(x=t, color="red", linestyle="--", alpha=0.5)
            ax.text(
                t,
                ax.get_ylim()[1] * 0.9,
                label,
                color="red",
                fontsize=8,
                rotation=90,
                ha="right",
            )

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path)
    plt.close()
