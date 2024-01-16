## Directory

```plaintext
Team_23/
├── data_retrieval/
│   ├── __init__.py
│   └── get_data.py  # Contains GetData class with create_retriever_object, get_user_input, query_database methods
├── preprocessing/
│   ├── __init__.py
│   ├── cleaning_reformat.py  # Contains CleaningReformat class with data_cleaning, data_reformatting methods
│   └── scale.py  # Contains Scale class with normalize_data, regularize_data methods
├── function_implementation/
│   ├── __init__.py
│   ├── function_handling.py  # Contains FunctionHandling class with load_custom_function method
│   ├── outlier_detection.py  # Contains OutlierDetection class with detect_outliers method
│   ├── function_validation.py  # Contains FunctionValidation class with validate_function method
│   └── get_stars.py # Contains GetStars class with get_good_stars_info, get_bad_stars_info methods 
├── model_fitting/
│   ├── __init__.py
│   ├── baseline_regression.py  # Contains BaselineRegression class with get_parameters, fit_model, predict_results methods (average prediction)
│   ├── logistic_regression.py  # Contains LogisticRegression class with get_parameters, set_parameters, fit_model, predict_results methods
│   └── linear_regression.py  # Contains LinearRegression class with get_parameters, fit_model, predict_results methods
├── visualization/
│   ├── __init__.py
│   ├── scatter_plot.py  # Contains ScatterPlot class with choose_features, draw_scatter methods
│   └── regression_plot.py  # Contains RegressionPlot class with draw_regression method
├── tests/
│   ├── test_data_retrieval.py
│   ├── test_preprocessing.py
│   ├── test_function_implementation.py
│   ├── test_model_fitting.py
│   └── test_visualization_tests.py
├── examples/
│   └── example_usage.py
├── setup.py
├── pyproject.toml
├── README.md
└── requirements.txt
```

## Test Suite
The test suite is located inside the `tests/` directory.

## Package Distribution
To distribute the package, use PyPI with setuptools. Prepare `setup.py` and `pyproject.toml` for PyPI distribution, ensuring all necessary information is included.

## Other Considerations
- **Documentation:** Add docstrings for classes and functions.
- **Unit Testing:** Consider adding more granular test files for each class.
- **Version Control:** Ensure there are no merge conflicts between different developers.

## Licensing
This code is released under the MIT License. You may freely use, build, or disseminate on this code as long as provide attribution back to this repository. We are not liable for any 
issues arising from the use of this code. 