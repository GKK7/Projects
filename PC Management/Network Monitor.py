# Import the required module
import psutil


def network_monitor():
    # Get network statistics
    net_io_counters = psutil.net_io_counters()

    # Get bytes sent
    bytes_sent = net_io_counters.bytes_sent

    # Get bytes received
    bytes_received = net_io_counters.bytes_recv

    # convert bytes to MB
    sent_mb = bytes_sent / (1024 * 1024)
    received_mb = bytes_received / (1024 * 1024)

    # Print the results
    print("Sent: {:.2f} MB".format(sent_mb))
    print("Received: {:.2f} MB".format(received_mb))


network_monitor()
