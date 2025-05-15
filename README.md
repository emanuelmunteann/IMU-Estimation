Human Motion Classification Using Random Forest and BNO-055 Sensor Data

📖 Overview
This repository tells the story of a comprehensive machine learning project dedicated to classifying human motions using data collected from three BNO-055 inertial measurement units (IMUs) strategically placed on the back, shoulder, and bicep. At its core, the project harnesses a Random Forest classifier to interpret complex patterns in orientation and acceleration, transforming raw sensor data into meaningful motion categories. The codebase is thoughtfully organized, starting with robust data exploration tools-such as `data.info()` and `data.describe(include='all')`-to summarize and describe the dataset’s structure and statistics. Advanced visualization techniques, including histograms, boxplots, and Q-Q plots, are employed to uncover hidden patterns and detect outliers, ensuring data quality and integrity. The analysis further comes to life with interactive 3D surface plots that illustrate how key performance metrics (RMSE and R²) for biceps movement vary across different model parameters, providing valuable insights into model behavior. Ultimately, the repository guides users through the entire workflow, from data loading and exploration to model training and testing, offering a powerful toolkit for anyone interested in human motion classification using IMU data.


## 🗃️ Dataset
### Sensor Configuration:
3 BNO-055 sensors positioned at:
  - Upper Back
  - Right Shoulder
  - Right Bicep
  
### Collected Data
Each sensor captures:
- Angles (X, Y, Z Axis)
  
### Data Structure
CSV files with columns
