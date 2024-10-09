import pandas as pd
import matplotlib.pyplot as plt

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

plot_voltage_vs_time(
    file_path='power_measurement/LED_Power_usage.csv',
    data_delimiter=',',
    skip_rows=7,
    x_label='Time (S)',
    y_label='Voltage (mV)',
    y_ticks=(0, 165, 5),
    y_lims=(0, 160),
    title='Difference of potential onto the 20 Ohm resistance without WFI and after LED is turned on',
    output_file='power_measurement/voltage_vs_time.png'
)
plot_voltage_vs_time(
    file_path='power_measurement/WithWFI_powerconsum.csv',
    data_delimiter=',',
    skip_rows=7,
    x_label='Time (S)',
    y_label='Voltage (mV)',
    y_ticks=(0, 50, 2),
    y_lims=(0, 50),
    title='Difference of potential onto the 20 Ohm resistance with WFI and after LED is turned on',
    output_file='power_measurement/voltage_vs_time_WFI.png'
)

def plot_power_vs_time(file_path, data_delimiter, skip_rows, x_label, y_label, y_ticks, y_lims, title, output_file):
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

    # Save the plot as an image file
    plt.savefig(output_file)

plot_power_vs_time(
    file_path='power_measurement/LED_Power_usage.csv',
    data_delimiter=',',
    skip_rows=7,
    x_label='Time (S)',
    y_label='Power (mW)',
    y_ticks=(0, 30, 1),
    y_lims=(0, 30),
    title='Power usage onto the 20 Ohm resistance without WFI and after LED is turned on',
    output_file='power_measurement/power_vs_time.png'
)
plot_power_vs_time(
    file_path='power_measurement/WithWFI_powerconsum.csv',
    data_delimiter=',',
    skip_rows=7,
    x_label='Time (S)',
    y_label='Power (mW)',
    y_ticks=(0, 10, 1),
    y_lims=(0, 10),
    title='Power usage onto the 20 Ohm resistance with WFI and after LED is turned on',
    output_file='power_measurement/power_vs_time_WFI.png'
)