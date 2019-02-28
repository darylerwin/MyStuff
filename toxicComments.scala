import com.bbmtek.utils.ETLUtils._
import com.bbmtek.schema.{BbmChannelsPostsSchema, BbmChannelsChannelSchema, UserProfileSchema, ChannelImageSchema, ProfileImageSchema}
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.functions._

spark.conf.set("spark.bbmtek.environment", "production-bigdata")

val trainingData = (spark.loadTable[BbmChannelsPostsSchema]("2019-02-10")
  .where('body.isNotNull and 'flag_count.isNotNull)
  .withColumn("elapsed_days", datediff(lit("2019-02-10"), 'created_at))
  .where('elapsed_days < 1500 and 'elapsed_days > 30)
  .where(('flag_count === 0 and 'elapsed_days < 44) or ('flag_count > 5))
  .withColumn("label", when('flag_count>0, 1).otherwise(0))
  .select('body, 'label)
)
trainingData.cache()
trainingData.show

trainingData.groupBy('label).count.show

// Feature extraction
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer, CountVectorizer, CountVectorizerModel}

val tokenizer = new Tokenizer().setInputCol("body").setOutputCol("words")
val wordsData = tokenizer.transform(trainingData)

val countVectorizer = new CountVectorizer().setInputCol("words").setOutputCol("rawFeatures").setVocabSize(2000000).fit(wordsData)
val featurizedData = countVectorizer.transform(wordsData)

//val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(2000000)
//val featurizedData = hashingTF.transform(wordsData)
// alternatively, CountVectorizer can also be used to get term frequency vectors

//featurizedData.show

val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
val idfModel = idf.fit(featurizedData)

val rescaledData = idfModel.transform(featurizedData)
//rescaledData.select("label", "features").show(false)

val vocabulary = countVectorizer.vocabulary

import org.apache.spark.ml.classification.LogisticRegression

val lr = (new LogisticRegression()
  .setMaxIter(10)
  .setRegParam(0.3)
  .setFeaturesCol("features")
  .setLabelCol("label")
)

// Fit the model
val lrModel = lr.fit(rescaledData)

// Print the coefficients and intercept for logistic regression
// println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

// Accuracy
import org.apache.spark.sql.types._
val predictions = (lrModel.transform(rescaledData)
  .select('prediction.cast(IntegerType), 'label)
)
predictions.show

val accuracy = predictions.withColumn("correct", ('prediction === 'label).cast(IntegerType))
accuracy.groupBy('correct).count.show

predictions.groupBy('label, 'prediction).count.show

// Get the actaul feature importances from the logistic regression model
// Get array of coefficients
val vocabulary = Array("(Intercept)") ++ countVectorizer.vocabulary
val coefficients = Array(lrModel.intercept) ++ lrModel.coefficients.toArray

val df = sc.parallelize(vocabulary zip coefficients).toDF("Word","Coefficients")
df.orderBy('Coefficients).show(10, false)
df.orderBy('Coefficients.desc).show(10, false)
//println("Feature\tCoefficient")
//vocabulary.zip(coefficients).foreach { case (feature, coeff) =>
//  println(s"$feature\coeff")
//}
