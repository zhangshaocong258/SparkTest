import java.util.HashMap;
import java.util.Map;

import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.SparkConf;

/**
 * Created by zsc on 2017/6/6.
 */
public class DecisionTreeClassificationExample {
    public static void main(String[] args) {
        final Integer finalLabelIndex = 14;
        final String delimiter = ",";

        SparkConf sparkConf = new SparkConf().setAppName("JavaDecisionTree");
        JavaSparkContext sc = new JavaSparkContext("local", "spark", sparkConf);

        // Load and parse the data file.
        String datapath = "data/mllib/Adult Census Income Binary Classification dataset1.csv";

        //普通txt文件，内容全部为double
//        JavaRDD<String> rddData = sc.textFile(datapath);
//        JavaRDD<LabeledPoint> data = rddData.map(
//                new Function<String, LabeledPoint>() {
//                    public LabeledPoint call(String line) {
//                        String[] parts = line.split(delimiter);
//                        double[] v = new double[parts.length - 1];
//                        for (int i = 0; i < parts.length - 1; i++) {
//                            v[i] = Double.parseDouble(parts[i]);
//                        }
//                        return new LabeledPoint(Double.parseDouble(parts[finalLabelIndex]), Vectors.dense(v));
//                    }
//                }
//        );
        JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(sc.sc(), datapath).toJavaRDD();
        // Split the data into training and test sets (30% held out for testing)
        JavaRDD<LabeledPoint>[] splits = data.randomSplit(new double[]{0.7, 0.3});
        JavaRDD<LabeledPoint> trainingData = splits[0];
        JavaRDD<LabeledPoint> testData = splits[1];

        // Set parameters.
        //  Empty categoricalFeaturesInfo indicates all features are continuous.
        Integer numClasses = 2;
        Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>();
        String impurity = "gini";
        Integer maxDepth = 5;
        Integer maxBins = 32;

        // Train a DecisionTree model for classification.
        final DecisionTreeModel model = DecisionTree.trainClassifier(trainingData, numClasses,
                categoricalFeaturesInfo, impurity, maxDepth, maxBins);

        // Evaluate model on test instances and compute test error
        JavaPairRDD<Double, Double> predictionAndLabel =
                testData.mapToPair(new PairFunction<LabeledPoint, Double, Double>() {
                    @Override
                    public Tuple2<Double, Double> call(LabeledPoint p) {
                        return new Tuple2<Double, Double>(model.predict(p.features()), p.label());
                    }
                });
        Double testErr =
                1.0 * predictionAndLabel.filter(new Function<Tuple2<Double, Double>, Boolean>() {
                    @Override
                    public Boolean call(Tuple2<Double, Double> pl) {
                        return !pl._1().equals(pl._2());
                    }
                }).count() / testData.count();
        System.out.println("Test Error: " + testErr);
        System.out.println("Learned classification tree model:\n" + model.toDebugString());

        // Save and load model
//        model.save(sc.sc(), "myModelPath");
//        DecisionTreeModel sameModel = DecisionTreeModel.load(sc.sc(), "myModelPath");
    }
}
