from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline

# Créer une session Spark
spark = SparkSession.builder.appName("AirfoilNoise").getOrCreate()

# Lire l'ensemble de données CSV
data = spark.read.csv("path/to/NASA_airfoil_noise_raw.csv", header=True, inferSchema=True)

# Préparation des données
data = data.withColumnRenamed("SoundLevel", "label")
assembler = VectorAssembler(inputCols=["Frequency", "Angle_of_Attack", "ChordLength", "FreeStreamVelocity", "SuctionSideDisplacement"], outputCol="features")

# Normalisation des données
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")

# Régression linéaire
lr = LinearRegression(featuresCol="scaledFeatures", labelCol="label")

# Création du pipeline
pipeline = Pipeline(stages=[assembler, scaler, lr])

# Entraînement du modèle
model = pipeline.fit(data)

# Extraction des coefficients
coefficients = model.stages[-1].coefficients
intercept = model.stages[-1].intercept

print(f"Coefficients: {coefficients}")
print(f"Intercept: {intercept}")

# Afficher le coefficient du déplacement côté aspiration
suction_side_displacement_coefficient = coefficients[4]  # Assuming it's the fifth feature
print(f"Coefficient de SuctionSideDisplacement: {suction_side_displacement_coefficient}")
