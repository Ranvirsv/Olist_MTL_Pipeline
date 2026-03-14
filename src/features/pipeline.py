from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from cyclic import CyclicEncoder
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline

def build_preprocessor():
    numeric_features = [
        'total_freight_value', 'product_weight_g', 
        'total_payment_value', 'max_installments', 'geo_distance_km'
    ]

    cat_features = [
        'product_category_name', 'seller_state', 'customer_state'
    ]

    temporal_features = [
        'order_purchase_timestamp'
    ]

    number_tranformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])

    cat_transformer = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num_transformer", number_tranformer, numeric_features),
            ("cat_transformer", cat_transformer, cat_features),
            ("time_transformer", CyclicEncoder(), temporal_features)
        ], remainder="drop"
    )

    return preprocessor

