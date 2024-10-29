#!/bin/bash

# Define the recipient email addresses
RECIPIENTS=("narayana.n@taoautomation.com" "chaitra.shivanagoudar@taoautomation.com" "hariharamoorthy.theriappan@taoautomation.com" "narendra.venkata@taoautomation.com" "rupesh.alluri@taoautomation.com" "sahil.aggarwal@taoautomation.com" "amit.rajan@taoautomation.com")

# Define the Kafka service name
KAFKA_SERVICE_NAME="kafka"

# Get Kafka service status using systemctl
KAFKA_SERVICE_STATUS=$(systemctl status "$KAFKA_SERVICE_NAME")

# Extract the running status of Kafka from the systemctl status output
KAFKA_RUNNING_STATUS=$(echo "$KAFKA_SERVICE_STATUS" | awk '/Active:/ {print $2}')

# Check if Kafka is active/running
if [[ "$KAFKA_RUNNING_STATUS" == "active" ]]; then
    KAFKA_TOPIC_STATUS="Kafka is running"
elif [[ "$KAFKA_RUNNING_STATUS" == "failed" ]]; then
    KAFKA_TOPIC_STATUS="Kafka is not running"
    SUBJECT_HIGH_IMPORTANCE="KAFKA DOWN RULE 1"
    # Define the email content with high importance for Kafka down alert
    EMAIL_BODY_KAFKA_DOWN="This is the email body for Kafka down alert"
    for recipient in "${RECIPIENTS[@]}"; do
        echo -e "$EMAIL_BODY_KAFKA_DOWN\n\nKafka is Down: Immediate Attention Required" | mail -s "$SUBJECT_HIGH_IMPORTANCE" -a 'X-Priority: 1 (Highest)' "$recipient"
    done

    cd /tmp/kafka-logs/
    rm -rf *
    systemctl restart kafka
    SUBJECT_HIGH_IMPORTANCE="KAFKA UP RULE 1"
    for recipient in "${RECIPIENTS[@]}"; do
        echo -e "$EMAIL_BODY_KAFKA_DOWN\n\nKafka is Up: Kafka restarted successfully" | mail -s "$SUBJECT_HIGH_IMPORTANCE" -a 'X-Priority: 1 (Highest)' "$recipient"
    done
else
    # Check Kafka service status again
    SERVICE_STATUS=$(systemctl is-active kafka)
    if [[ "$SERVICE_STATUS" == "failed" ]]; then
        SUBJECT_HIGH_IMPORTANCE="KAFKA NOT UP RULE 1"
        EMAIL_BODY_KAFKA_DOWN="This is the email body for Kafka down alert"
        for recipient in "${RECIPIENTS[@]}"; do
            echo -e "$EMAIL_BODY_KAFKA_DOWN\n\nKafka Down: Failed to restart Kafka" | mail -s "$SUBJECT_HIGH_IMPORTANCE" -a 'X-Priority: 1 (Highest)' "$recipient"
        done
    fi
fi

# Define the email subject
SUBJECT="Blinkit Rule 1 Server"
CPU_CORES=$(grep -c processor /proc/cpuinfo)
CPU_USAGE=$(top -b -n 1 | awk 'NR>7{s+=$9}END{printf "%.2f\n", s / '$CPU_CORES' }')
# Check CPU, RAM, and disk usage
#CPU_USAGE=$(top -b -n 1 | awk 'NR>7{s+=$9}END{print s}')
RAM_USAGE=$(free | awk '/Mem/{printf("%.2f"), $3/$2*100}')
DISK_USAGE=$(df / | awk 'NR==2{print $5}' | tr -d '%')

# Threshold for resource usage
THRESHOLD=95
cpu_threshold=120
# Prepare email body for CPU, RAM, and disk usage
EMAIL_BODY_RESOURCE="CPU: $CPU_USAGE%\nRAM: $RAM_USAGE%\nDISK: $DISK_USAGE%"

# Check if any resource usage exceeds the threshold
if [ $(echo "$CPU_USAGE > $cpu_threshold" | bc -l) -eq 1 ] || \
   [ $(echo "$RAM_USAGE > $THRESHOLD" | bc -l) -eq 1 ] || \
   [ $(echo "$DISK_USAGE > $THRESHOLD" | bc -l) -eq 1 ]; then
    # Send an email for resource usage alert with Kafka status
    EMAIL_BODY_RESOURCE_KAFKA="$EMAIL_BODY_RESOURCE\n\nKafka Status:\n$KAFKA_TOPIC_STATUS"
    for recipient in "${RECIPIENTS[@]}"; do
        echo -e "$EMAIL_BODY_RESOURCE_KAFKA\n\nResource Usage Alert: CPU, RAM, or Disk exceeds threshold" | mail -s "$SUBJECT" "$recipient"
    done
fi

