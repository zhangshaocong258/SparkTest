import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.mllib.fpm.AssociationRules;
import org.apache.spark.mllib.fpm.FPGrowth;
import org.apache.spark.mllib.fpm.FPGrowthModel;
import scala.Tuple2;

/**
 * Created by zsc on 2017/6/2.
 */
public class FPGrowthExample {
    public static void main(String args[]) {
        SparkConf sparkConf = new SparkConf().setAppName("FP-Growth");
        JavaSparkContext sc = new JavaSparkContext("local", "spark", sparkConf);

        JavaRDD<String> data = sc.textFile("data/mllib/sample_fpgrowth2.txt");

        JavaRDD<List<String>> transactions = data.map(
                new Function<String, List<String>>() {
                    public List<String> call(String line) {
                        String[] parts = line.split(" ");
                        return Arrays.asList(parts);
                    }
                }
        );

        FPGrowth fpg = new FPGrowth().setMinSupport(0.3).setNumPartitions(10);
        FPGrowthModel<String> model = fpg.run(transactions);

        for (FPGrowth.FreqItemset<String> itemset : model.freqItemsets().toJavaRDD().collect()) {
            System.out.println("[" + itemset.javaItems() + "], " + itemset.freq());
        }

        double minConfidence = 0.6;
        for (AssociationRules.Rule<String> rule
                : model.generateAssociationRules(minConfidence).toJavaRDD().collect()) {
            System.out.println(
                    rule.javaAntecedent() + " => " + rule.javaConsequent() + ", " + rule.confidence());
        }


        /**
         * java版本generateAssociationRules
         */
        final double finalminConfidence = minConfidence;
        //所有频繁项集
        JavaRDD<FPGrowth.FreqItemset<String>> freqItemsets = model.freqItemsets().toJavaRDD();

        // For candidate rule X => Y, generate (X, (Y, freq(X union Y))) Y长度为1
        JavaPairRDD<List<String>, FPGrowth.FreqItemset<String>> candidates = freqItemsets
                .flatMapToPair(new PairFlatMapFunction<FPGrowth.FreqItemset<String>,
                        List<String>, FPGrowth.FreqItemset<String>>() {
                    @Override
                    public Iterable<Tuple2<List<String>, FPGrowth.FreqItemset<String>>> call(FPGrowth.FreqItemset<String> stringFreqItemset) throws Exception {
                        List<Tuple2<List<String>, FPGrowth.FreqItemset<String>>> l1 = new ArrayList<Tuple2<List<String>, FPGrowth.FreqItemset<String>>>();
                        List<Tuple2<List<String>, FPGrowth.FreqItemset<String>>> l2 = new ArrayList<Tuple2<List<String>, FPGrowth.FreqItemset<String>>>();

                        List<String> items = stringFreqItemset.javaItems();

                        if (items.size() != 1) {
                            for (int i = 0; i < items.size(); i++) {
                                List<String> consequent = new ArrayList<String>();
                                List<String> antecedent = new ArrayList<String>();

                                consequent.clear();
                                antecedent.clear();
                                consequent.add(items.get(i));
                                //items.remove(i);
                                for (int j = 0; j < items.size(); j++) {
                                    if (j != i) {
                                        antecedent.add(items.get(j));
                                    }
                                }

                                FPGrowth.FreqItemset<String> freqItemset2 = new FPGrowth.FreqItemset(consequent.toArray(),
                                        stringFreqItemset.freq());
                                l1.add(new Tuple2<List<String>, FPGrowth.FreqItemset<String>>(antecedent, freqItemset2));
                            }
                            return l1;

                        }
                        return l2;

                    }
                });
        System.out.println("candidates" + candidates.count());

        // generate all (X, freq(X))
        JavaPairRDD<List<String>, Double> modeldata = freqItemsets.flatMapToPair(new PairFlatMapFunction<FPGrowth.FreqItemset<String>,
                List<String>, Double>() {
            @Override
            public Iterable<Tuple2<List<String>, Double>> call(FPGrowth.FreqItemset<String> stringFreqItemset) throws Exception {
                List<Tuple2<List<String>, Double>> l = new ArrayList<Tuple2<List<String>, Double>>();
                l.add(new Tuple2<List<String>, Double>(stringFreqItemset.javaItems(), (double) stringFreqItemset.freq()));
                return l;
            }
        });
        System.out.println("modeldata" + modeldata.count());

        // Join to get (X, ((Y, freq(X union Y)), freq(X)))
        JavaPairRDD<List<String>, Tuple2<FPGrowth.FreqItemset<String>, Double>> finaldata = candidates.join(modeldata);

        //generate rules 输入(X, ((Y, freq(X union Y)), freq(X)))
        JavaPairRDD<List<List<String>>, Double> result = finaldata.flatMapToPair(new PairFlatMapFunction<Tuple2<List<String>,
                Tuple2<FPGrowth.FreqItemset<String>, Double>>, List<List<String>>, Double>() {
            @Override
            public Iterable<Tuple2<List<List<String>>, Double>> call(Tuple2<List<String>, Tuple2<FPGrowth.FreqItemset<String>,
                    Double>> listTuple2Tuple2) throws Exception {

                List<Tuple2<List<List<String>>, Double>> l1 = new ArrayList<Tuple2<List<List<String>>, Double>>();
                List<List<String>> l2 = new ArrayList<List<String>>();
                List<String> antecedent = new ArrayList<String>();//antecedent
                List<String> consequent = new ArrayList<String>();//consequent
                Double confidence = null;

                antecedent = listTuple2Tuple2._1();
                consequent = listTuple2Tuple2._2()._1().javaItems();
                confidence = listTuple2Tuple2._2()._1().freq() / listTuple2Tuple2._2()._2();
                l2.add(antecedent);
                l2.add(consequent);
                l1.add(new Tuple2<List<List<String>>, Double>(l2, confidence));
                return l1;
            }
        });
        System.out.println("result" + result.count());

        //filter by confidence
        JavaPairRDD<List<List<String>>, Double> finalresult = result.filter(new Function<Tuple2<List<List<String>>, Double>, Boolean>() {
            @Override
            public Boolean call(Tuple2<List<List<String>>, Double> v1) throws Exception {
                Boolean t = v1._2() >= finalminConfidence;
                return t;
            }
        });
        System.out.println(("finalresult" + finalresult.count()));

        List<Tuple2<List<List<String>>, Double>> finallist = finalresult.collect();
        for (int i = 0; i < finallist.size(); i++) {
            System.out.println((finallist.get(i)._1().get(0) + "=>" + finallist.get(i)._1().get(1) + "  " + finallist.get(i)._2()));
        }
    }
}
