# Trichotillomania Awareness Project 🚨

## Overview

Welcome to the Trichotillomania Awareness Project! This project aims to detect and warn individuals with Trichotillomania, a disorder characterized by the urge to pull out one's hair, particularly from the scalp, eyebrows, and eyelashes. The system utilizes computer vision techniques and Python to detect facial hair pulling behavior in real-time. As someone who suffers with this condition, this idea came as a potential solution with the use of emerging powerful technologies.

See the application in action:

![Trichotillomania Awareness Project](demo.gif)

## Table of Contents

- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)

## Getting Started 

To get started with the Trichotillomania Awareness Project, follow the instructions in this README. This project is designed to raise awareness and provide real-time warnings to individuals engaging in hair-pulling behavior.

## Prerequisites

Before running the project, make sure you have the following prerequisites installed:

- Python 
- OpenCV
- Mediapipe
- NumPy
- Simpleaudio
- Streamlit

Install the required Python packages using the following command:

``` bash
pip install opencv-python numpy mediapipe simpleaudio streamlit
```

## Installation

Clone the repository to your local machine:

``` bash
git clone https://github.com/miguelarmada/Trichotillomania-Vision.git
```

## Usage 

Run the main python script with streamlit to start using the application:

``` bash
streamlit run main-app.py
```

The script will use your default camera to perform real-time tracking of hands and facial landmarks. With this information, a sound will play when the system detects hair-pulling behaviour, bringing awareness to this situation.

## Contributing 

If you have ideas for improvements or new features, please submit a pull request or contact me to discuss proposed changes.
