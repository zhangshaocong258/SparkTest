import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.*;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.tree.configuration.FeatureType;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.mllib.tree.model.Node;
import org.apache.spark.api.java.JavaPairRDD;

import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.model.Split;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;


import java.util.*;
import java.text.DecimalFormat;

/**
 * Created by zsc on 2017/6/6.
 */
public class MyDecisionTreeClassificationExample {

    private static final String DELIMITER = ",";

    private static final String Separator = "_";

    private static final String CLASSIFICATION = "classification";

    private static final String REGRESSION = "regression";

    private static DecimalFormat decimalFormat = new DecimalFormat("#0.000000");

    private static String input = "data/mllib/Adult Census Income Binary Classification dataset1.csv";    //数据输入路径

    private static String algo = CLASSIFICATION;     //训练算法：分类或回归

    private static Integer maxDepth = 5;  //树最大深度

    private static Integer maxBins = 2;    //树最大分类个数

    private static Integer labelIndex = 14;    //标签索引

    private static Integer numClasses = -1;    //类个数

    private static String impurity = "gini";

    private static int depth = 5;

    private static HashMap<Integer, String>[] reverseMap;

    private static HashMap<String, Integer>[] featureMap;

    private static List<String> features;

    private static String head = "age,workclass,fnlwgt,education,educationnum,maritalstatus,occupation,relationship," +
            "race,sex,capitalgain,capitalloss,hoursperweek,nativecountry,income";

    public static void main(String[] args) {

        SparkConf sparkConf = new SparkConf().setAppName("JavaDecisionTree");
        JavaSparkContext sc = new JavaSparkContext("local", "spark", sparkConf);

        // Load and parse the data file.
        String datapath = input;
        JavaRDD<String> dataWithHead = sc.textFile(datapath);
        head = dataWithHead.first();
        features = Arrays.asList(head.split(","));
        labelIndex = features.size() - 1;

        JavaRDD<String> data = dataWithHead.filter(new Function<String, Boolean>() {
            @Override
            public Boolean call(String str) throws Exception {
                return !str.equals(head);
            }
        }).persist(StorageLevel.MEMORY_AND_DISK());
        //计算数据维度
        int dimension = data.take(1).get(0).split(DELIMITER).length;

        // Set parameters.
        //  Empty categoricalFeaturesInfo indicates all features are continuous.
        HashMap<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>();

        HashMap<String, Integer>[] StringFuture2Int = new HashMap[dimension];//每一列，<标签，数字>，数字递增，且各不相同
        for (int i = 0; i < StringFuture2Int.length; i++) {
            StringFuture2Int[i] = new HashMap<String, Integer>();
        }

        //分别获取数据每一维的离散数据并映射为数值
        final Integer finalLabelIndex = labelIndex;
        final String finalAlgo = algo;

        /**
         * begin
         */
        /*JavaPairRDD<String, Integer> futures = data.flatMapToPair(new PairFlatMapFunction<String, String, Integer>() {
            @Override
            public Iterable<Tuple2<String, Integer>> call(String s) throws Exception {
                ArrayList<Tuple2<String, Integer>> arrayList = new ArrayList<>();
                String[] temp = s.split(DELIMITER);
                for (int i = 0; i < temp.length; i++) {
                    if (finalAlgo.equalsIgnoreCase(CLASSIFICATION) && i == finalLabelIndex) {
                        arrayList.add(new Tuple2<String, Integer>(new StringBuilder().append(i).append(Separator).append(temp[i]).toString(), 1));
                        continue;
                    } else if (finalAlgo.equalsIgnoreCase(REGRESSION) && i == finalLabelIndex) {
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
        futures.saveAsTextFile("data/mllib/futures");

        try {
            //会去重
            JavaPairRDD<String, Integer> distinctFutures = futures.reduceByKey(new Function2<Integer, Integer, Integer>() {
                @Override
                public Integer call(Integer v1, Integer v2) throws Exception {
                    return 1;
                }
            });
            distinctFutures.saveAsTextFile("data/mllib/distinctFutures");

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

            List<Tuple2<Integer, Iterable<String>>> listOfFutures = reduceFuture.collect();*/
        /**
         * end
         */

        /**
         * begin
         */
        JavaRDD<String> features = data.flatMap(new FlatMapFunction<String, String>() {
            @Override
            public Iterable<String> call(String s) throws Exception {
                ArrayList<String> arrayList = new ArrayList<>();
                String[] temp = s.split(DELIMITER);
                for (int i = 0; i < temp.length; i++) {
                    if (finalAlgo.equalsIgnoreCase(CLASSIFICATION) && i == finalLabelIndex) {
                        arrayList.add(new StringBuilder().append(i).append(Separator).append(temp[i]).toString());
                        continue;
                    } else if (finalAlgo.equalsIgnoreCase(REGRESSION) && i == finalLabelIndex) {
                        try {
                            Double.valueOf(temp[i]);
                        } catch (NumberFormatException e) {
                            arrayList.add(new StringBuilder().append(i).append(Separator).append(temp[i]).toString());
                        }
                    }
                    try {
                        Double.valueOf(temp[i]);
                    } catch (Exception e) {
                        arrayList.add(new StringBuilder().append(i).append(Separator).append(temp[i]).toString());
                    }
                }
                if (arrayList.size() != 0) {
                    return arrayList;
                } else {
                    return null;
                }            }
        });
        features.saveAsTextFile("data/mllib/features");


        try {
            JavaRDD<String> distinctFeatures = features.distinct();
            distinctFeatures.saveAsTextFile("data/mllib/distinctFeatures");

            JavaPairRDD<Integer, String> dimensionFeatures = distinctFeatures.mapToPair(new PairFunction<String, Integer, String>() {
                @Override
                public Tuple2<Integer, String> call(String s) throws Exception {
                    String[] key = s.split(Separator);
                    if (key.length == 2) {
                        return new Tuple2<>(Integer.valueOf(key[0]), key[1]);
                    }
                    return null;                }
            });
            JavaPairRDD<Integer, Iterable<String>> reduceFeature = dimensionFeatures.groupByKey();

            List<Tuple2<Integer, Iterable<String>>> listOfFutures = reduceFeature.collect();
            /**
             * end
             */

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

        featureMap = StringFuture2Int;

        //将输入数据处理成算法所需格式
        final HashMap<String, Integer>[] hashMap = StringFuture2Int;
        JavaRDD<LabeledPoint> parsedData = data.map(
                new Function<String, LabeledPoint>() {
                    public LabeledPoint call(String line) {
                        String[] parts = line.split(DELIMITER);
                        double[] v = new double[parts.length - 1];
                        double label = 0.0D;
                        try {
                            if (finalAlgo.equalsIgnoreCase(CLASSIFICATION)) {
                                label = hashMap[hashMap.length - 1].get(parts[finalLabelIndex]);
                            } else if (finalAlgo.equalsIgnoreCase(REGRESSION)) {
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
            @Override
            public Double call(LabeledPoint p) {
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

        HashMap<Integer, String> labelMap = new HashMap<>();

        //获取标签特征映射关系
        for (Map.Entry<String, Integer> entry : hashMap[hashMap.length - 1].entrySet()) {
            labelMap.put(entry.getValue(), entry.getKey());
        }

        //对输入数据进行预测
        final HashMap<Integer, String> finalLabelMap = labelMap;
        final Broadcast<HashMap<Integer, String>> finalBro = sc.broadcast(labelMap);

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
                        if (finalBro.value().size() != 0) {
                            label = finalBro.value().get(predict.intValue());
                        }

                        //输出结果
                        StringBuffer outputLine = new StringBuffer();
                        if (finalRightLabel != -1) {
                            outputLine.append(finalBro.value().get((int) (labeledPoint.label())));
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
                     * @return Array[0]:result
                     *         Array[1]:probility
                     */
                    private double[] predict(Node node, Vector vector) {
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
//        List<String> finalResult = result.collect();
//        for (String str : finalResult) {
//            System.out.println(str);
//        }
//        result.saveAsTextFile("data/mllib/result.csv");

        /**
         * 显示图
         */
        reverseFeatureMap();
        toDebugString(model);
        System.out.println("*******************");

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

    private static void toDebugString(DecisionTreeModel model) {
        ArrayList<String> arrayList = new ArrayList<>();
        int count = 0;

        if (algo.equalsIgnoreCase(CLASSIFICATION)) {
            System.out.println(String.format("DecisionTreeModel classifier of depth %d depth with %d numNodes nodes",
                    model.topNode().subtreeDepth(), model.topNode().numDescendants() + 1));
        }

        System.out.println(subtreeToString(model.topNode(), arrayList, count));
    }


    private static String subtreeToString(Node top, ArrayList<String> arrayList, int indentFactor) {
        String prefix = genPrefix(indentFactor);
        if (top.isLeaf()) {
            if (reverseMap[reverseMap.length - 1] != null && !reverseMap[reverseMap.length - 1].isEmpty()) {
                return (prefix + "Predict " + reverseMap[reverseMap.length - 1].get((int) top.predict().predict()) + "\n");
            } else {
                return (prefix + "Predict " + top.predict().predict() + "\n");
            }
        } else {
            return (prefix + "If " + splitToString(top.split().get(), true) + "\n" +
                    subtreeToString(top.leftNode().get(), arrayList, indentFactor + 1) +
                    prefix + "Else " + splitToString(top.split().get(), false) + "\n" +
                    subtreeToString(top.rightNode().get(), arrayList, indentFactor + 1));
        }
    }

    private static String splitToString(Split split, boolean left) {
        if (left) {
            if (reverseMap[split.feature()] != null && !reverseMap[split.feature()].isEmpty()) {
                scala.collection.Iterator<Object> list = split.categories().iterator();
                StringBuilder stringBuilder = new StringBuilder();
                while (list.hasNext()) {
                    stringBuilder.append(reverseMap[split.feature()].get(Double.valueOf(list.next().toString()).intValue()));
                    stringBuilder.append(",");
                }
                //迭代器被通过后，size length 均为0，即迭代器智能通过一次
                if (stringBuilder.length() != 0) {
                    stringBuilder.deleteCharAt(stringBuilder.length() - 1);
                }
                return String.format("(feature %s in %s)", features.get(split.feature()), stringBuilder.toString());
            }
            return String.format("(feature %s <= %s)", features.get(split.feature()), split.threshold());
        } else {
            if (reverseMap[split.feature()] != null && !reverseMap[split.feature()].isEmpty()) {
                scala.collection.Iterator<Object> list = split.categories().iterator();
                StringBuilder stringBuilder = new StringBuilder();
                while (list.hasNext()) {
                    stringBuilder.append(reverseMap[split.feature()].get(Double.valueOf(list.next().toString()).intValue()));
                    stringBuilder.append(",");
                }
                if (stringBuilder.length() != 0) {
                    stringBuilder.deleteCharAt(stringBuilder.length() - 1);
                }
                return String.format("(feature %s not in %s )", features.get(split.feature()), stringBuilder.toString());
            }
            return String.format("(feature %s > %s )", features.get(split.feature()), split.threshold());
        }
    }

    private static String genPrefix(int indentFactor) {

        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < indentFactor; i++) {
            sb.append("\t");
        }
        return sb.toString();
    }


    private static void reverseFeatureMap() {
        reverseMap = new HashMap[featureMap.length];
        for (int i = 0; i < reverseMap.length; i++) {
            reverseMap[i] = new HashMap<>();
            for (Map.Entry<String, Integer> entry : featureMap[i].entrySet()) {
                reverseMap[i].put(entry.getValue(), entry.getKey());
            }
        }
    }


    private static double predict(Node node, Vector vector) {
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
