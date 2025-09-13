from typing import Dict, List, Tuple, Any
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def split_train_validation(
        raw_df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the raw dataframe into training and validation sets.

    Args:
        raw_df (pd.DataFrame): The raw input dataframe
        test_size (float, optional): Proportion of the dataset to include in the validation split.
                                   Defaults to 0.2.
        random_state (int, optional): Random state for reproducibility. Defaults to 42.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Training and validation dataframes

    Raises:
        KeyError: If 'Exited' column is not found in the dataframe
        ValueError: If test_size is not between 0 and 1
    """
    if 'Exited' not in raw_df.columns:
        raise KeyError("Column 'Exited' not found in dataframe")

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    train_df, val_df = train_test_split(
        raw_df,
        test_size=test_size,
        random_state=random_state,
        stratify=raw_df['Exited']
    )

    return train_df, val_df


def separate_features_targets(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str], str]:
    """
    Separate input features and target variables from training and validation dataframes.

    Args:
        train_df (pd.DataFrame): Training dataframe
        val_df (pd.DataFrame): Validation dataframe

    Returns:
        Tuple containing:
            - train_inputs (pd.DataFrame): Training input features
            - train_targets (pd.Series): Training target values
            - val_inputs (pd.DataFrame): Validation input features
            - val_targets (pd.Series): Validation target values
            - input_cols (List[str]): List of input column names
            - target_col (str): Target column name
    """
    input_cols = list(train_df.columns[:-1])
    target_col = train_df.columns[-1]

    train_inputs = train_df[input_cols].copy()
    train_targets = train_df[target_col].copy()
    val_inputs = val_df[input_cols].copy()
    val_targets = val_df[target_col].copy()

    return train_inputs, train_targets, val_inputs, val_targets, input_cols, target_col


def identify_numeric_columns(
        train_inputs: pd.DataFrame,
        exclude_cols: List[str] = None
) -> List[str]:
    """
    Identify numeric columns in the training inputs, excluding specified columns.

    Args:
        train_inputs (pd.DataFrame): Training input features dataframe
        exclude_cols (List[str], optional): Columns to exclude from numeric columns.
                                          Defaults to ['CustomerId', 'id'].

    Returns:
        List[str]: List of relevant numeric column names
    """
    if exclude_cols is None:
        exclude_cols = ['CustomerId', 'id']

    numeric_cols = train_inputs.select_dtypes(include='number').columns.tolist()
    numeric_cols_relevant = [col for col in numeric_cols if col not in exclude_cols]

    return numeric_cols_relevant


def scale_numeric_features(
        train_inputs: pd.DataFrame,
        val_inputs: pd.DataFrame,
        numeric_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """
    Scale numeric features using MinMaxScaler fitted on training data.

    Args:
        train_inputs (pd.DataFrame): Training input features dataframe
        val_inputs (pd.DataFrame): Validation input features dataframe
        numeric_cols (List[str]): List of numeric columns to scale

    Returns:
        Tuple containing:
            - train_inputs (pd.DataFrame): Training inputs with scaled numeric features
            - val_inputs (pd.DataFrame): Validation inputs with scaled numeric features
            - scaler (MinMaxScaler): Fitted scaler object

    Raises:
        KeyError: If any column in numeric_cols is not found in the dataframes
    """
    # Verify all numeric columns exist
    missing_cols = [col for col in numeric_cols if col not in train_inputs.columns]
    if missing_cols:
        raise KeyError(f"Columns not found in training data: {missing_cols}")

    scaler = MinMaxScaler()
    scaler.fit(train_inputs[numeric_cols])

    # Create copies to avoid modifying original dataframes
    train_scaled = train_inputs.copy()
    val_scaled = val_inputs.copy()

    train_scaled[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
    val_scaled[numeric_cols] = scaler.transform(val_inputs[numeric_cols])

    return train_scaled, val_scaled, scaler


def encode_gender_feature(
        train_inputs: pd.DataFrame,
        val_inputs: pd.DataFrame,
        gender_mapping: Dict[str, int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Encode gender feature using manual mapping.

    Args:
        train_inputs (pd.DataFrame): Training input features dataframe
        val_inputs (pd.DataFrame): Validation input features dataframe
        gender_mapping (Dict[str, int], optional): Mapping for gender encoding.
                                                 Defaults to {'Female': 0, 'Male': 1}.

    Returns:
        Tuple containing:
            - train_inputs (pd.DataFrame): Training inputs with gender encoded
            - val_inputs (pd.DataFrame): Validation inputs with gender encoded

    Raises:
        KeyError: If 'Gender' column is not found in the dataframes
    """
    if gender_mapping is None:
        gender_mapping = {'Female': 0, 'Male': 1}

    if 'Gender' not in train_inputs.columns:
        raise KeyError("Column 'Gender' not found in training data")

    # Create copies to avoid modifying original dataframes
    train_encoded = train_inputs.copy()
    val_encoded = val_inputs.copy()

    train_encoded['Gender_Code'] = train_encoded['Gender'].map(gender_mapping)
    val_encoded['Gender_Code'] = val_encoded['Gender'].map(gender_mapping)

    return train_encoded, val_encoded


def encode_categorical_features(
        train_inputs: pd.DataFrame,
        val_inputs: pd.DataFrame,
        categorical_cols: List[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder, List[str]]:
    """
    One-hot encode categorical features using encoder fitted on training data.

    Args:
        train_inputs (pd.DataFrame): Training input features dataframe
        val_inputs (pd.DataFrame): Validation input features dataframe
        categorical_cols (List[str], optional): Columns to encode. Defaults to ['Geography'].

    Returns:
        Tuple containing:
            - train_inputs (pd.DataFrame): Training inputs with encoded categorical features
            - val_inputs (pd.DataFrame): Validation inputs with encoded categorical features
            - encoder (OneHotEncoder): Fitted encoder object
            - encoded_cols (List[str]): List of new encoded column names

    Raises:
        KeyError: If any categorical column is not found in the dataframes
    """
    if categorical_cols is None:
        categorical_cols = ['Geography']

    # Verify all categorical columns exist
    missing_cols = [col for col in categorical_cols if col not in train_inputs.columns]
    if missing_cols:
        raise KeyError(f"Columns not found in training data: {missing_cols}")

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(train_inputs[categorical_cols])

    # Create copies to avoid modifying original dataframes
    train_encoded = train_inputs.copy()
    val_encoded = val_inputs.copy()

    # Get encoded column names
    encoded_cols = []
    for i, col in enumerate(categorical_cols):
        encoded_cols.extend(list(encoder.categories_[i]))

    # Transform and add encoded columns
    train_encoded_features = encoder.transform(train_inputs[categorical_cols])
    val_encoded_features = encoder.transform(val_inputs[categorical_cols])

    # Add encoded features as new columns
    for i, col_name in enumerate(encoded_cols):
        train_encoded[col_name] = train_encoded_features[:, i]
        val_encoded[col_name] = val_encoded_features[:, i]

    return train_encoded, val_encoded, encoder, encoded_cols


def select_final_features(
        train_inputs: pd.DataFrame,
        val_inputs: pd.DataFrame,
        numeric_cols: List[str],
        encoded_cols: List[str],
        additional_cols: List[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Select final feature columns for modeling.

    Args:
        train_inputs (pd.DataFrame): Training input features dataframe
        val_inputs (pd.DataFrame): Validation input features dataframe
        numeric_cols (List[str]): List of numeric column names
        encoded_cols (List[str]): List of encoded categorical column names
        additional_cols (List[str], optional): Additional columns to include.
                                            Defaults to ['Gender_Code'].

    Returns:
        Tuple containing:
            - X_train (pd.DataFrame): Final training feature matrix
            - X_val (pd.DataFrame): Final validation feature matrix
    """
    if additional_cols is None:
        additional_cols = ['Gender_Code']

    final_cols = numeric_cols + encoded_cols + additional_cols

    # Verify all columns exist
    missing_train = [col for col in final_cols if col not in train_inputs.columns]
    missing_val = [col for col in final_cols if col not in val_inputs.columns]

    if missing_train:
        raise KeyError(f"Columns not found in training data: {missing_train}")
    if missing_val:
        raise KeyError(f"Columns not found in validation data: {missing_val}")

    X_train = train_inputs[final_cols].copy()
    X_val = val_inputs[final_cols].copy()

    return X_train, X_val


def preprocess_new_data(
        new_data: pd.DataFrame,
        scaler: MinMaxScaler,
        encoder: OneHotEncoder,
        input_cols: List[str] = None,
        exclude_cols: List[str] = None,
        categorical_cols: List[str] = None,
        gender_mapping: Dict[str, int] = None,
        additional_cols: List[str] = None
) -> pd.DataFrame:
    """
    Preprocess new data using already fitted scaler and encoder.

    This function is designed to handle new data (like test.csv) using the same
    preprocessing steps as the training data, but with already fitted transformers.

    Args:
        new_data (pd.DataFrame): New data to preprocess
        scaler (MinMaxScaler): Already fitted MinMaxScaler from training
        encoder (OneHotEncoder): Already fitted OneHotEncoder from training
        input_cols (List[str], optional): Input columns to select. If None, uses all columns.
        exclude_cols (List[str], optional): Columns to exclude from scaling.
                                          Defaults to ['CustomerId', 'id'].
        categorical_cols (List[str], optional): Categorical columns to encode.
                                              Defaults to ['Geography'].
        gender_mapping (Dict[str, int], optional): Gender encoding mapping.
                                                 Defaults to {'Female': 0, 'Male': 1}.
        additional_cols (List[str], optional): Additional columns to include in final features.
                                             Defaults to ['Gender_Code'].

    Returns:
        pd.DataFrame: Preprocessed feature matrix ready for model prediction

    Raises:
        KeyError: If required columns are missing from new_data
        ValueError: If scaler or encoder are not fitted

    Example:
        >>> # After training
        >>> result = preprocess_data_pipeline(train_df)
        >>> scaler = result['scaler']
        >>> encoder = result['encoder']
        >>> input_cols = result['input_cols']
        >>>
        >>> # For new test data
        >>> X_test = preprocess_new_data(test_df, scaler, encoder, input_cols)
    """
    # Set defaults
    if exclude_cols is None:
        exclude_cols = ['CustomerId', 'id']
    if categorical_cols is None:
        categorical_cols = ['Geography']
    if gender_mapping is None:
        gender_mapping = {'Female': 0, 'Male': 1}
    if additional_cols is None:
        additional_cols = ['Gender_Code']

    # Validate that transformers are fitted
    try:
        # Check if scaler is fitted by accessing its attributes
        _ = scaler.scale_
    except AttributeError:
        raise ValueError("Scaler is not fitted. Please fit the scaler on training data first.")

    try:
        # Check if encoder is fitted by accessing its attributes
        _ = encoder.categories_
    except AttributeError:
        raise ValueError("Encoder is not fitted. Please fit the encoder on training data first.")

    # Step 1: Select input columns if specified
    if input_cols is not None:
        # Check if all input columns are present
        missing_cols = [col for col in input_cols if col not in new_data.columns]
        if missing_cols:
            raise KeyError(f"Input columns not found in new data: {missing_cols}")
        processed_data = new_data[input_cols].copy()
    else:
        processed_data = new_data.copy()

    # Step 2: Identify and scale numeric features
    numeric_cols_relevant = identify_numeric_columns(processed_data, exclude_cols)

    # Verify numeric columns exist
    missing_numeric = [col for col in numeric_cols_relevant if col not in processed_data.columns]
    if missing_numeric:
        raise KeyError(f"Numeric columns not found in new data: {missing_numeric}")

    # Apply scaling using fitted scaler
    processed_data[numeric_cols_relevant] = scaler.transform(processed_data[numeric_cols_relevant])

    # Step 3: Encode gender feature
    if 'Gender' not in processed_data.columns:
        raise KeyError("Column 'Gender' not found in new data")

    processed_data['Gender_Code'] = processed_data['Gender'].map(gender_mapping)

    # Check for unmapped gender values
    if processed_data['Gender_Code'].isnull().any():
        unmapped_genders = processed_data[processed_data['Gender_Code'].isnull()]['Gender'].unique()
        raise ValueError(f"Unknown gender values found: {unmapped_genders}. "
                         f"Expected values: {list(gender_mapping.keys())}")

    # Step 4: Encode categorical features using fitted encoder
    missing_categorical = [col for col in categorical_cols if col not in processed_data.columns]
    if missing_categorical:
        raise KeyError(f"Categorical columns not found in new data: {missing_categorical}")

    # Get encoded column names from fitted encoder
    encoded_cols = []
    for i, col in enumerate(categorical_cols):
        encoded_cols.extend(list(encoder.categories_[i]))

    # Transform categorical features using fitted encoder
    encoded_features = encoder.transform(processed_data[categorical_cols])

    # Add encoded features as new columns
    for i, col_name in enumerate(encoded_cols):
        processed_data[col_name] = encoded_features[:, i]

    # Step 5: Select final feature columns
    final_cols = numeric_cols_relevant + encoded_cols + additional_cols

    # Verify all final columns exist
    missing_final = [col for col in final_cols if col not in processed_data.columns]
    if missing_final:
        raise KeyError(f"Final feature columns not found in processed data: {missing_final}")

    X_processed = processed_data[final_cols].copy()

    return X_processed


def preprocess_data(
        raw_df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42
) -> Dict[str, Any]:
    """
    Complete data preprocessing pipeline that orchestrates all preprocessing steps.

    This function should be called instead of the original preprocess_data function.
    It performs the following steps:
    1. Split data into train/validation sets
    2. Separate features and targets
    3. Scale numeric features
    4. Encode categorical features (gender and geography)
    5. Select final feature set

    Args:
        raw_df (pd.DataFrame): Raw input dataframe with all features and target
        test_size (float, optional): Proportion for validation split. Defaults to 0.2.
        random_state (int, optional): Random state for reproducibility. Defaults to 42.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - 'X_train': Training feature matrix
            - 'train_targets': Training target values
            - 'X_val': Validation feature matrix
            - 'val_targets': Validation target values
            - 'input_cols': List of original input column names
            - 'scaler': Fitted MinMaxScaler object
            - 'encoder': Fitted OneHotEncoder object

    Raises:
        KeyError: If required columns ('Exited', 'Gender', 'Geography') are missing
        ValueError: If test_size is not between 0 and 1
    """
    # Step 1: Split data
    train_df, val_df = split_train_validation(raw_df, test_size, random_state)

    # Step 2: Separate features and targets
    train_inputs, train_targets, val_inputs, val_targets, input_cols, _ = (
        separate_features_targets(train_df, val_df)
    )

    # Step 3: Identify and scale numeric features
    numeric_cols_relevant = identify_numeric_columns(train_inputs)
    train_inputs, val_inputs, scaler = scale_numeric_features(
        train_inputs, val_inputs, numeric_cols_relevant
    )

    # Step 4: Encode gender feature
    train_inputs, val_inputs = encode_gender_feature(train_inputs, val_inputs)

    # Step 5: Encode categorical features (Geography)
    train_inputs, val_inputs, encoder, encoded_cols = encode_categorical_features(
        train_inputs, val_inputs
    )

    # Step 6: Select final features
    X_train, X_val = select_final_features(
        train_inputs, val_inputs, numeric_cols_relevant, encoded_cols
    )

    return {
        'X_train': X_train,
        'train_targets': train_targets,
        'X_val': X_val,
        'val_targets': val_targets,
        'input_cols': input_cols,
        'scaler': scaler,
        'encoder': encoder
    }
