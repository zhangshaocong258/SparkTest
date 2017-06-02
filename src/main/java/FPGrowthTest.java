import java.util.Arrays;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.fpm.AssociationRules;
import org.apache.spark.mllib.fpm.FPGrowth;
import org.apache.spark.mllib.fpm.FPGrowthModel;

/**
 * Created by zsc on 2017/6/2.
 */
public class FPGrowthTest {
    public static void main(String args[]) {
        SparkConf sparkConf = new SparkConf().setAppName("FP-Growth");
        JavaSparkContext sc = new JavaSparkContext("local", "spark", sparkConf);

        JavaRDD<String> data = sc.textFile("data/mllib/sample_fpgrowth.txt");

        JavaRDD<List<String>> transactions = data.map(
                new Function<String, List<String>>() {
                    public List<String> call(String line) {
                        String[] parts = line.split(" ");
                        return Arrays.asList(parts);
                    }
                }
        );

        FPGrowth fpg = new FPGrowth().setMinSupport(0.2).setNumPartitions(10);
        FPGrowthModel<String> model = fpg.run(transactions);

        for (FPGrowth.FreqItemset<String> itemset : model.freqItemsets().toJavaRDD().collect()) {
            System.out.println("[" + itemset.javaItems() + "], " + itemset.freq());
        }

        double minConfidence = 0.8;
        for (AssociationRules.Rule<String> rule
                : model.generateAssociationRules(minConfidence).toJavaRDD().collect()) {
            System.out.println(
                    rule.javaAntecedent() + " => " + rule.javaConsequent() + ", " + rule.confidence());
        }
    }
}
