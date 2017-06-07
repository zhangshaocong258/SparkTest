import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.tree.configuration.FeatureType;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.mllib.tree.model.Node;
import org.apache.spark.api.java.JavaPairRDD;

import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;


import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.text.DecimalFormat;
import java.util.Map;

/**
 * Created by zsc on 2017/6/6.
 */
public class MyDecisionTreeClassificationExample {

    private static final String DELIMITER = ",";

    private static final String Separator = "_";

    private static final String CLASSIFICATION = "classification";

    private static final String REGRESSION = "regression";

    private static DecimalFormat decimalFormat = new DecimalFormat("#0.000000");


    public static void main(String[] args) {
        String input = "data/mllib/Adult Census Income Binary Classification dataset1.csv";    //数据输入路径
        String algo = CLASSIFICATION;     //训练算法：分类或回归
        Integer maxDepth = 5;  //树最大深度
        Integer maxBins = 2;    //树最大分类个数
        Integer labelIndex = 14;    //标签索引
        Integer numClasses = -1;    //类个数
        String impurity = "gini";

        SparkConf sparkConf = new SparkConf().setAppName("JavaDecisionTree");
        JavaSparkContext sc = new JavaSparkContext("local", "spark", sparkConf);

        // Load and parse the data file.
        String datapath = input;

        JavaRDD<String> data = sc.textFile(datapath).persist(StorageLevel.MEMORY_AND_DISK());

        //计算数据维度
        int dimension = data.take(1).get(0).split(DELIMITER).length;

        // Set parameters.
        //  Empty categoricalFeaturesInfo indicates all features are continuous.
        HashMap<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>();

        HashMap<String, Integer>[] StringFuture2Int = new HashMap[dimension];//每一列，<标签，数字>，数字递增，且各不相同
        for (int i = 0; i < StringFuture2Int.length; i++) {
            StringFuture2Int[i] = new HashMap<>();
        }

        //分别获取数据每一维的离散数据并映射为数值
        final Integer finalLabelIndex = labelIndex;
        final String finalAlgo = algo;
        JavaPairRDD<String, Integer> futures = data.flatMapToPair(new PairFlatMapFunction<String, String, Integer>() {
            @Override
            public Iterable<Tuple2<String, Integer>> call(String s) throws Exception {
                ArrayList<Tuple2<String, Integer>> arrayList = new ArrayList<>();
                String[] temp = s.split(DELIMITER);
                for (int i = 0; i < temp.length; i++) {
                    if(finalAlgo.equalsIgnoreCase(CLASSIFICATION) && i == finalLabelIndex){
                        arrayList.add(new Tuple2<String, Integer>(new StringBuilder().append(i).append(Separator).append(temp[i]).toString(), 1));
                        continue;
                    }else if(finalAlgo.equalsIgnoreCase(REGRESSION) && i == finalLabelIndex){
                        try {
                            Double.valueOf(temp[i]);
                        } catch (NumberFormatException e) {
                            arrayList.add(new Tuple2<String, Integer>(new StringBuilder().append(i).append(Separator).append(temp[i]).toString(), 1));
                        }
                    }
                    try {
                        Double.valueOf(temp[i]);
                    } catch (Exception e) {
                        arrayList.add(new Tuple2<String, Integer>(new StringBuilder().append(i).append(Separator).append(temp[i]).toString(), 1));
                    }
                }
                if (arrayList.size() != 0) {
                    return arrayList;
                } else {
                    return null;
                }
            }
        });

        try {
            JavaPairRDD<String, Integer> distinctFutures = futures.reduceByKey(new Function2<Integer, Integer, Integer>() {
                @Override
                public Integer call(Integer v1, Integer v2) throws Exception {
                    return 1;
                }
            });

            JavaPairRDD<Integer, String> dimensionFutures = distinctFutures.mapToPair(new PairFunction<Tuple2<String, Integer>, Integer, String>() {
                @Override
                public Tuple2<Integer, String> call(Tuple2<String, Integer> stringIntegerTuple2) throws Exception {
                    String[] key = stringIntegerTuple2._1().split(Separator);
                    if (key.length == 2) {
                        return new Tuple2<>(Integer.valueOf(key[0]), key[1]);
                    }
                    return null;
                }
            });

            JavaPairRDD<Integer, Iterable<String>> reduceFuture = dimensionFutures.groupByKey();

            List<Tuple2<Integer, Iterable<String>>> listOfFutures = reduceFuture.collect();

            //将映射关系存入HashMap及categoricalFeaturesInfo
            for (int i = 0; i < listOfFutures.size(); i++) {
                Iterator<String> iter = listOfFutures.get(i)._2().iterator();
                int index = listOfFutures.get(i)._1();
                int futureSize = 0;
                int futureNum = 0;
                while (iter.hasNext()) {
                    String s = iter.next();
                    if (StringFuture2Int[index].containsKey(s)) {
                        continue;
                    } else {
                        futureSize++;
                        StringFuture2Int[index].put(s, futureNum++);
                    }
                }
                //只有一个值，那么一列是废的
                if (futureSize > 1) {
                    if (index == labelIndex) {
                        continue;
                    } else {
                        if (index > labelIndex) {
                            index -= 1;
                        }
                        categoricalFeaturesInfo.put(index, futureSize);
                    }
                }
            }
            //把标签列移到最后
            HashMap<String, Integer> temp = StringFuture2Int[labelIndex];
            for (int i = labelIndex; i < StringFuture2Int.length - 1; i++) {
                StringFuture2Int[i] = StringFuture2Int[i + 1];
            }
            StringFuture2Int[StringFuture2Int.length - 1] = temp;
        } catch (Exception e) {
            System.out.println("all futures are continues");
        }

        for (int i = 0; i < StringFuture2Int.length; i++) {
            int size = StringFuture2Int[i].size();
            if (maxBins < size) {
                maxBins = size;
            }
        }

        //将输入数据处理成算法所需格式
        final HashMap<String, Integer>[] hashMap = StringFuture2Int;
        JavaRDD<LabeledPoint> parsedData = data.map(
                new Function<String, LabeledPoint>() {
                    public LabeledPoint call(String line) {
                        String[] parts = line.split(DELIMITER);
                        double[] v = new double[parts.length - 1];
                        double label = 0.0D;
                        try {
                            if(finalAlgo.equalsIgnoreCase(CLASSIFICATION)) {
                                label = hashMap[hashMap.length - 1].get(parts[finalLabelIndex]);
                            }else if(finalAlgo.equalsIgnoreCase(REGRESSION)){
                                try {
                                    label = Double.parseDouble(parts[finalLabelIndex]);
                                } catch (NumberFormatException e) {
                                    try {
                                        label = hashMap[hashMap.length - 1].get(parts[finalLabelIndex]);
                                    } catch (Exception ne) {
                                        System.out.println("program error, impossible here!");
                                    }
                                }
                            }
                        } catch (Exception ne) {
                            System.out.println("impossible here , program error");
                        }
                        int j = 0;
                        for (int i = 0; i < parts.length; i++) {
                            if (i == finalLabelIndex) {
                                continue;
                            }
                            try {
                                v[j] = Double.parseDouble(parts[i]);
                            } catch (NumberFormatException e) {
                                try {
                                    v[j] = hashMap[j].get(parts[i]);
                                } catch (Exception ne) {
                                    System.out.println("program error, impossible here!");
                                }
                            }
                            j++;
                        }
                        return new LabeledPoint(label, Vectors.dense(v));
                    }
                }
        ).persist(StorageLevel.DISK_ONLY());

        // Compute the number of classes from the data.
        numClasses = parsedData.map(new Function<LabeledPoint, Double>() {
            @Override public Double call(LabeledPoint p) {
                return p.label();
            }
        }).countByValue().size();


        // Split the data into training and test sets (30% held out for testing)
        JavaRDD<LabeledPoint>[] splits = parsedData.randomSplit(new double[]{0.7, 0.3});
        JavaRDD<LabeledPoint> trainingData = splits[0];
        JavaRDD<LabeledPoint> testData = splits[1].persist(StorageLevel.MEMORY_AND_DISK());


        // Train a DecisionTree model for classification.
        final DecisionTreeModel model = DecisionTree.trainClassifier(trainingData, numClasses,
                categoricalFeaturesInfo, impurity, maxDepth, maxBins);

        /**
         * 预测
         */

        int rightLabelIndex = 14;   //正确标签索引
        int idIndex = -1;           //id索引


        HashMap<Integer, String> labelMap = new HashMap<>();

        //获取标签特征映射关系
        for(Map.Entry<String, Integer> entry : hashMap[hashMap.length - 1].entrySet()){
            labelMap.put(entry.getValue(), entry.getKey());
        }

        //对输入数据进行预测
        final HashMap<Integer, String> finalLabelMap = labelMap;
        final int finalRightLabel = rightLabelIndex;
        System.out.println("labelMap" + labelMap.toString());
        final String delimiter = ",";
        JavaRDD<String> result = testData.map(
                new Function<LabeledPoint, String>() {
                    public String call(LabeledPoint labeledPoint) {

                        //得到预测结果及相应的转换
                        double[] result = predict(model.topNode(), labeledPoint.features());
//                        Double predict = model.predict(Vectors.dense(v));
                        Double predict = result[0];
                        String label = String.valueOf(predict);
                        if (finalLabelMap.size() != 0) {
                            label = finalLabelMap.get(predict.intValue());
                        }

                        //输出结果
                        StringBuffer outputLine = new StringBuffer();
                        if (finalRightLabel != -1) {
                            outputLine.append(finalLabelMap.get((int) (labeledPoint.label())));
                            outputLine.append(delimiter);
                        }

                        //预测概率
                        double probability = Math.abs(result[1]);
                        outputLine.append(label).append(delimiter).append(decimalFormat.format(probability));
                        return outputLine.toString();
                    }

                    /**
                     * 决策树模型predict方法
                     *
                     * @param node : 模型根节点
                     * @param vector : 预测输入
                     *
                     * @return  Array[0]:result
                     *          Array[1]:probility
                     * */
                    private double[] predict (Node node, Vector vector) {
                        if (node.isLeaf()) {
                            double[] result = new double[2];
                            result[0] = node.predict().predict();
                            result[1] = node.predict().prob();
                            return result;
                        } else {
                            if (node.split().get().featureType() == FeatureType.Continuous()) {
                                if (vector.apply(node.split().get().feature()) <= node.split().get().threshold()) {
                                    return predict(node.leftNode().get(), vector);
                                } else {
                                    return predict(node.rightNode().get(), vector);
                                }
                            } else {
                                if (node.split().get().categories().contains(vector.apply(node.split().get().feature()))) {
                                    return predict(node.leftNode().get(), vector);
                                } else {
                                    return predict(node.rightNode().get(), vector);
                                }
                            }
                        }
                    }
                }
        );

        //得到结果 正确结果，预测结果，概率
        List<String> finalResult = result.collect();
        for (String str : finalResult) {
            System.out.println(str);
        }


        // Evaluate model on test instances and compute test error
        JavaPairRDD<Double, Double> predictionAndLabel =
                testData.mapToPair(new PairFunction<LabeledPoint, Double, Double>() {
                    @Override
                    public Tuple2<Double, Double> call(LabeledPoint p) {
                        return new Tuple2<Double, Double>(predict(model.topNode(), p.features()), p.label());
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
        System.out.println("Learned classification tree model:\n" + model.toDebugString());//

        // Save and load model
//        model.save(sc.sc(), "myModelPath");
//        DecisionTreeModel sameModel = DecisionTreeModel.load(sc.sc(), "myModelPath");
    }

    private static String toDebugString(DecisionTreeModel model) {

        return "";
    }

    private static double predict (Node node, Vector vector) {
        if (node.isLeaf()) {
            double result = node.predict().predict();
            return result;
        } else {
            if (node.split().get().featureType() == FeatureType.Continuous()) {
                if (vector.apply(node.split().get().feature()) <= node.split().get().threshold()) {
                    return predict(node.leftNode().get(), vector);
                } else {
                    return predict(node.rightNode().get(), vector);
                }
            } else {
                if (node.split().get().categories().contains(vector.apply(node.split().get().feature()))) {
                    return predict(node.leftNode().get(), vector);
                } else {
                    return predict(node.rightNode().get(), vector);
                }
            }
        }
    }
}
