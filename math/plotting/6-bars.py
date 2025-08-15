#!/usr/bin/env python3
"""
6-bars.py
"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """
    Documented
    """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))
    persons = ['Farrah', 'Fred', 'Felicia']

    plt.legend()
    plt.title('Number of Fruit per Person')
    plt.ylabel('Quantity of Fruit')
    plt.xticks(np.arange(3), persons)
    plt.yticks(np.arange(0, 81, 10))
    plt.ylim(0, 80)
    plt.bar(np.arange(3), fruit[0], color='red',
            width=0.5, label='apples')
    plt.bar(np.arange(3), fruit[1], color='yellow',
            width=0.5, label='bananas', bottom=fruit[0])
    plt.bar(np.arange(3), fruit[2], color='#ff8000',
            width=0.5, label='oranges', bottom=fruit[0] + fruit[1])
    plt.bar(np.arange(3), fruit[3], color='#ffe5b4',
            width=0.5, label='peaches', bottom=fruit[0] + fruit[1] + fruit[2])
    plt.legend(loc='upper right')
    plt.show()
