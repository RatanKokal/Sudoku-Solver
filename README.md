# Vision Based Sudoku Solver

## Important stages in the project
- Preprocessing the image to get only the sudoku in the image, so that 81 cells can be extracted easily
- Predicting the digits in each cell using machine learning
- Solving the sudoku using backtracking

## Dataset used for training tensorflow based CNN model
[Kaggle Dataset](https://www.kaggle.com/datasets/kshitijdhama/printed-digits-dataset/data) which boosted the accuracy to great extent and is much more useful than MNIST in this case

## Guide to run
- For solving sudoku: ```python solve.py```
<img src="./assets/readme/solve_help.png" alternate = "Solve.py help" width="600">

- For training model: ```python model_train.py train```
<img src="./assets/readme/train_help.png" alternate = "model_train.py train help" width="600">

- For evaluating model: ```python model_train.py evaluate```
<img src="./assets/readme/evaluate_help.png" alternate = "model_train.py evaluate help" width="600">

## Model Summary
<img src="./assets/readme/model_summary.png" alternate = "Model Summary" width="400">

## Example of a run

- Command: `python solve.py --image_path ./assets/sudoku_images/sudoku.jpg --model_path ./assets/models/model.keras`
- Sudoku image:
<img src="./assets/sudoku_images/sudoku.jpg" alternate = "Unsolved Sudoku" width="400">
- Terminal result:
<img src="./assets/readme/terminal.png" alternate = "Terminal" width="600">
