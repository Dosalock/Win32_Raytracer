
<img src="images/balls.png" width="700">
 
# My Raytracer Project

## Description

This project is a basic raytracer implemented in C++. It renders a simple scene consisting of spheres and lights, generating images from the tracing of rays through a virtual scene. This project showcases how a basic raytracer can work, allowing one to render simple scenes with diffuse lighting.

## Features

*   **Sphere Rendering:**  Renders spheres within the virtual scene.
*   **Basic Lighting:** Implements diffuse lighting model using directionnal, point and ambient lighting. 
*   **Reflections:** Able to render reflections recursively with a depth set in the Draw() function.
*   **Command-Line Execution:** Renders a scene from a console application.

## Getting Started

These instructions will help you set up and run the raytracer on your local machine.

### Prerequisites

*   C++ compiler (e.g., g++, MSVC)
*   Make sure you have installed an IDE like VSCode or Visual Studio

### Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/Dosalock/Win32_Raytracer.git
    cd src
    ```

2.  Compile the code:

*   **Using Visual Studio:** Open the solution file, and compile the project.
*   **Using VS Code:** Compile from the terminal.

    ```bash
        g++ main.cpp raytrace.cpp raystructs.cpp -o raytracer -lgdi32
    ```
   Note, you may need to add the gdi32 flag if your on windows

## Usage

Run the compiled executable:

```bash
./Window
```

This will render a simple scene, provide simple movement of camera (W,A,S,D, & Q,E).

## Project Structure

Here's a brief overview of the project structure:

```
My Project/
├── main.cpp           # Main application entry point and window setup.
├── main.h             # Header file for main.cpp.
├── raytrace.cpp       # Ray tracing implementation (ray calculations, scene rendering).
├── raytrace.h         # Header file for raytrace.cpp.
├── raystructs.cpp     # Currently unused 
├── raystructs.h       # Header file for data structures like Vect3D, Sphere, Light.
└── README.md          # This file.
```

## Class Documentation

*   **Light:** Represents a light source in the scene with properties such as type, intensity, and position.
*   **QuadraticAnswer:** A struct to hold the quadratic answer for solving the ray-sphere intersection.
*   **Sphere:** Represents a sphere object with center, radius, and color.
*   **Vect3D:** Represents a 3D vector with functions for vector calculations (dot product, length, normalization, etc.).

## Files Description
*   **main.cpp:** Contains the main application loop, window creation, and calls the raytracing routines.
*   **main.h:** Contains the headers required for the main file.
*   **raytrace.cpp:** Contains all of the raytracing, lighting and draw related functions.
*   **raytrace.h:** Contains the headers required for the raytrace file.
*   **raystructs.h:** Contains the struct for the Vect3D, Sphere, QuadraticAnswer and Light structs.

## Built With

*   C++

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

MIT License

Copyright (c) 2025 Johan Karlsson

## Contact

Johan Karlsson - j.ef.karlsson@gmail.com