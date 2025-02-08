+++
title = "Understanding Cooperative Rebalancing in Apache Kafka"
description = "In this post I explore Incremental Cooperative Rebalancing in Kafka"
categories =  [
	"distributed-systems"
]
date = "2020-09-04"
author = "Anirudh Ganesh"
[taxonomies]
tags = [
    "data-engineering",
    "data-processing",
    "distributed-systems",
    "kafka",
    "data-pipelines",
    "apache-kafka",
    "stream-processing",
    "message-queues",
    "kafka-rebalancing",
    "incremental-cooperative-rebalancing",
    "consumer-groups",
    "event-driven-architecture",
    "cloud-computing",
    "real-time-data",
    "scalability"
]

[extra]
toc = true
katex = true
+++

{% alert(caution=true) %}
* Diagrams
* Change some of the bullet points to organic paragraphs by rewriting it
* Add personal anecdote about copycat and how consumergroups are created
{%end%}

Apache Kafka is a distributed messaging system that handles large data volumes efficiently and reliably. Producers publish data to topics, and consumers subscribe to these topics to consume the data. It also stores data across multiple brokers and partitions, allowing it to scale horizontally.

One of the key challenges in maintaining Kafka's performance is ensuring a balanced distribution of partitions across consumers. Kafka's rebalancing process helps achieve this, but traditional approaches have significant drawbacks, leading to the introduction of incremental cooperative rebalancing as a more efficient alternative.

## Key Concepts in Kafka Rebalancing

Before diving into cooperative rebalancing, let's review some essential Kafka concepts:

<figure>
{{ image(url="kafka_architecture.svg", alt="A Brief Overview of Kafka Architecture") }}
<figcaption>
    A Brief Overview of Kafka Architecture
</figcaption>
</figure>

- **Brokers**: Servers that manage the storage and transfer of data within Kafka clusters.
- **Partition**: A unit of parallelism within a Kafka topic, where each partition is an ordered, immutable sequence of records.
- **Consumer Group**: A set of consumers that collaboratively consume data from one or more topics, with each consumer being assigned partitions.
- **Rebalance**: The process of redistributing partitions among consumer instances in a group.
- **Partition Assignment Strategy**: The algorithm Kafka uses to assign partitions, including round-robin, range, and sticky assignments.

## The Traditional Rebalancing Problem

Kafka rebalancing is triggered when consumers join or leave a group, new partitions are added, or consumer failures occur. The traditional "stop-the-world" rebalancing process involves:

1. Kafka notifying all consumers to stop consuming data.
2. Consumers rejoining the group and undergoing partition reassignment.
3. Consumers resuming processing once the rebalance completes.

This approach comes with its fair share of challenges.
One major issue is increased latency—whenever rebalancing happens, consumers are temporarily paused, this obviously leads to delays in processing messages.
This can create bottlenecks, especially in systems that rely on real-time data flow. In my case the consumer service would start firing availability alerts on our pipelines if it took longer than 15minutes.
On some very high throughput topics, this would mean to increase the consumer group size temporarily to handle the extra load this causes, then scale it back. Overall baby sitting this was unnecessary burden.

Another concern is reduced throughput.
Frequent rebalancing can disrupt the consumer group assignments, sometimes overloading some brokers and other times leaving them underutilized. This instability can make it harder to maintain optimal usage.

Also, there's the problem of higher resource usage.
Every time partitions are moved around, it consumes CPU, memory, and network bandwidth.
In large-scale deployments like ours, these overheads can quickly add up, impacting overall system performance.
Finally, there’s the risk of potential data loss.
If rebalancing isn’t handled carefully, unprocessed messages might slip through the cracks, leading to inconsistencies and gaps in the data pipeline, affecting some of our completeness guarantees.
Ensuring that messages aren’t lost during transitions is another challenge for maintaining our SLAs.

## Challenges with Stop-the-World Rebalancing

When I first started working with Kafka, I quickly ran into the quirks of its traditional rebalancing model.
On paper, it made sense—partitions get redistributed to keep things balanced.
But in reality, the process wasn’t as smooth as I had hoped, especially as the system grew in complexity.

One of the biggest pain points was scaling up and down. As we added more consumers or removed them, the rebalancing process became increasingly expensive.
The more resources there were to shuffle around, the longer everything took, making what should have been a simple operation a performance headache.

Then there was multi-tenancy.
In shared environments where different teams were running diverse workloads, a single rebalancing event could unexpectedly impact someone else’s pipeline.
I remember a time when our team’s batch processing job was disrupted mid-run simply because the consumer counts got adjusted on an unrelated topic.

Rolling upgrades were another tricky part.
Ideally, updating our infrastructure shouldn’t disrupt workloads, but Kafka’s default rebalancing model didn’t play nicely with rolling restarts.
I remember this one time, an upgrade window which caused a full rebalancing event, slowing things down so much that what should have been a minor version bump turned into a whole bunch of pages constantly buzzing me.

## Incremental Cooperative Rebalancing

To mitigate these issues, Kafka 2.4 introduced incremental cooperative rebalancing, which improves the rebalance process by allowing consumers to retain their current assignments while incrementally taking on new partitions. This approach minimizes disruptions and ensures a smoother transition.

## Key Improvements of Incremental Cooperative Rebalancing

- **Incremental Partition Transfer**: Only affected partitions are reassigned instead of stopping all consumers.
- **Continuous Data Processing**: Consumers remain active and process messages during rebalance.
- **Parallel Rebalancing**: New consumers can join and consume data concurrently.
- **Graceful Partition Handling**: Consumers are given a grace period to return before partitions are reassigned.
- **Sticky Assignor Implementation**: This strategy optimizes assignments by trying to retain previous partition allocations.

## How Incremental Cooperative Rebalancing Works

Kafka clients use an embedded rebalance protocol to manage partition assignments autonomously without burdening Kafka brokers. The rebalancing process follows these steps:

1. Kafka sends a `GroupCoordinator` message to notify consumers.
2. Consumers respond with a `JoinGroup` message to indicate participation.
3. Consumers voluntarily release partitions that need reassignment.
4. Incremental reassignment happens in multiple rounds without halting all consumers.

The process completes when the workload is evenly distributed across consumers.

## Reducing Rebalance Frequency

While rebalancing is necessary, frequent rebalances can degrade performance. Some best practices to reduce rebalancing events include:

- **Increase Session Timeout**: Adjust session.`timeout.ms` to allow consumers more time to recover before being marked as inactive.
- **Optimize Poll Intervals**: Set `max.poll.interval.ms` to accommodate longer processing times.
- **Limit Partitions per Topic**: Reducing the number of partitions can decrease rebalance occurrences.
- **Use Static Group Membership**: Assign partitions manually to consumers for a fixed workload distribution.

## Performance Improvements with Incremental Cooperative Rebalancing

Benchmarks[^1] have shown that incremental cooperative rebalancing drastically reduces rebalancing time and improves throughput in large-scale deployments. For instance, a test running 900 Kafka Connect tasks showed the following improvements:

- Aggregate throughput increased by **113%**.
- Median throughput improved by **101%**.
- Maximum throughput saw an **833% increase**.

These improvements result from the ability of Kafka clients to absorb resource fluctuations gracefully without completely stopping operations.

## Conclusion

Incremental cooperative rebalancing in Kafka is a significant step forward in improving consumer group management. By transitioning from the traditional stop-the-world approach to a more gradual, cooperative model, organizations can achieve higher availability, scalability, and performance in their Kafka deployments.

Implementing these techniques and understanding best practices ensures a more resilient and efficient streaming data pipeline, reducing disruptions and optimizing resource utilization.

[^1]: [Incremental Cooperative Rebalancing in Apache Kafka: Why Stop the World When You Can Change It?](https://www.confluent.io/blog/incremental-cooperative-rebalancing-in-kafka/)
