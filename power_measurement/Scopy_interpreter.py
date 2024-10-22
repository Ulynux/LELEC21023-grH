import pandas as pd
import matplotlib.pyplot as plt



# Times with led

time1_led = 1.825
time2_led = 2.8
time3_led = 7.047

# Times without led

time1_noled = 2.514
time2_noled = 3.491
time3_noled = 7.741


def plot_voltage_vs_time(file_path, data_delimiter, skip_rows, x_label, y_label, y_ticks, y_lims, title, output_file):
    data = pd.read_csv(file_path, delimiter=data_delimiter, skiprows=skip_rows)

    # Convert voltage to millivolts
    data['CH1(V)'] = data['CH1(V)'] * 1000

    plt.figure(figsize=(10, 6))
    plt.plot(data['Time(S)'], data['CH1(V)'], label='DOP (mV)')

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()

    plt.grid(True)

    # Ensure the y-axis includes zero and adjust the top limit if necessary
    plt.ylim(bottom=y_lims[0], top=y_lims[1])
    plt.yticks(range(y_ticks[0], y_ticks[1], y_ticks[2]))

    # Save the plot as an image file
    plt.savefig(output_file)

# plot_voltage_vs_time(
#     file_path='power_measurement/LED_Power_usage.csv',
#     data_delimiter=',',
#     skip_rows=7,
#     x_label='Time (S)',
#     y_label='Voltage (mV)',
#     y_ticks=(0, 165, 5),
#     y_lims=(0, 160),
#     title='Difference of potential onto the 20 Ohm resistance without WFI and after LED is turned on',
#     output_file='power_measurement/voltage_vs_time.png'
# )
# plot_voltage_vs_time(
#     file_path='power_measurement/WithWFI_powerconsum.csv',
#     data_delimiter=',',
#     skip_rows=7,
#     x_label='Time (S)',
#     y_label='Voltage (mV)',
#     y_ticks=(0, 50, 2),
#     y_lims=(0, 50),
#     title='Difference of potential onto the 20 Ohm resistance with WFI and after LED is turned on',
#     output_file='power_measurement/voltage_vs_time_WFI.png'
# )

def plot_power_vs_time(file_path, data_delimiter, skip_rows, x_label, y_label, y_ticks, y_lims, title, output_file, times):
    data = pd.read_csv(file_path, delimiter=data_delimiter, skiprows=skip_rows)

    # Convert voltage to power usage
    data['Power(W)'] = (data['CH1(V)'] * 1000 / 20) * 3.3

    plt.figure(figsize=(10, 6))
    plt.plot(data['Time(S)'], data['Power(W)'], label='Power (mW)')

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()

    plt.grid(True)

    # Ensure the y-axis includes zero and adjust the top limit if necessary
    plt.ylim(bottom=y_lims[0], top=y_lims[1])
    plt.yticks(range(y_ticks[0], y_ticks[1], y_ticks[2]))
    # Add vertical lines at the specified times
    for time in times:
        plt.axvline(x=time, color='r', linestyle='--')

    # Save the plot as an image file
    plt.savefig(output_file)

def piechart_power(file_path, data_delimiter, skip_rows, title, output_file, times):
    data = pd.read_csv(file_path, delimiter=data_delimiter, skiprows=skip_rows)

    # Convert voltage to power usage
    data['Power(W)'] = (data['CH1(V)'] / 20) * 3.3
    times = [time * 1000 for time in times]  # Scale of time step, convert to indices
    time1 = int(times[0])
    time2 = int(times[1])
    time3 = int(times[2])

    
    # Calculate the power usage for each segment
    print(data['Power(W)'][time1+20], data['Power(W)'][time2+20], data['Power(W)'][time3+20])
    power1 = data['Power(W)'][time1:time2].sum()   # * 0.001 (time step) *1000 (conversion to mW)
    power2 = data['Power(W)'][time2:time3].sum()   # * 0.001 (time step) *1000 (conversion to mW)
    power_values = [power2, power1]
    total_power = power1 + power2


    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))


    def func(pct):
        absolute = pct/100.*total_power
        return f"{pct:.1f}%\n({absolute:.2f} mJ)"


    wedges, texts, autotexts = ax.pie(power_values, autopct=lambda pct: func(pct),
                                    textprops=dict(color="w"))

    ax.legend(wedges, ["Subtask 2", "Subtask 1"],
            title=f"Tasks ({total_power:.2f} mJ)",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1))

    plt.setp(autotexts, size=8, weight="bold")
    ax.set_title(title)

    # Save the plot as an image file
    plt.savefig(output_file)

plot_power_vs_time(
    file_path='power_measurement/data_H2B.csv',
    data_delimiter=',',
    skip_rows=7,
    x_label='Time (S)',
    y_label='Power (mW)',
    y_ticks=(0, 20, 5),
    y_lims=(0, 20),
    title='Power usage of the audio sampling cycle with LED',
    output_file='power_measurement/power_audio_sampling_led.png',
    times=[time1_led, time2_led, time3_led]
)
plot_power_vs_time(
    file_path='power_measurement/data_H2B_sansled_V2.csv',
    data_delimiter=',',
    skip_rows=7,
    x_label='Time (S)',
    y_label='Power (mW)',
    y_ticks=(0, 20, 5),
    y_lims=(0, 20),
    title='Power usage of the audio sampling cycle without LED',
    output_file='power_measurement/power_audio_sampling_sansled.png',
    times=[time1_noled, time2_noled, time3_noled]
)


piechart_power(
    file_path="power_measurement/data_H2B.csv",
    data_delimiter=',',
    skip_rows=7,
    title='Power usage of the audio duty cycle with LED',
    output_file='power_measurement/piechart_audio_sampling_led.png',
    times=[time1_led, time2_led, time3_led]
)
piechart_power(
    file_path="power_measurement/data_H2B_sansled_V2.csv",
    data_delimiter=',',
    skip_rows=7,
    title='Power usage of the audio duty cycle without LED',
    output_file='power_measurement/piechart_audio_sampling_sansled.png',
    times=[time1_noled, time2_noled, time3_noled]
)