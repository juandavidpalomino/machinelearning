package machinelearning;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.List;


public class Example2Weka {

    public static void main(String[] args) throws Exception {

        List<Double[]> featuresList = new ArrayList<>();
        featuresList.add(new Double[] { 90.0 });
        featuresList.add(new Double[] { 101.0 });
        featuresList.add(new Double[] { 103.0 });
        featuresList.add(new Double[] { 90.0 });

        // create the labels
        List<Double> labels = new ArrayList<>();
        labels.add(249.0);
        labels.add(338.0);
        labels.add(304.0);


        // load the labels and features
        labels = ZData.loadLabels("/example2.txt");
        featuresList = ZData.loadFeaturesList("/example2.txt");


        // build the features list
        ArrayList<Attribute> attributes = new ArrayList<>();
        Attribute att1 = new Attribute("metros");
        attributes.add(att1);

        Attribute att2 = new Attribute("habitaciones");
        attributes.add(att2);

        Attribute att3 = new Attribute("banos");
        attributes.add(att3);

        Attribute att4 = new Attribute("garajes");
        attributes.add(att4);

        Attribute priceAttribute = new Attribute("priceLabel");
        attributes.add(priceAttribute);


        Instances trainingSet = new Instances("trainData", attributes, 10);
        trainingSet.setClassIndex(trainingSet.numAttributes() - 1);

        Instance instance = new DenseInstance(5);
        instance.setValue(att1, 100.0);
        instance.setValue(att2, 1.0);
        instance.setValue(att3, 2.0);
        instance.setValue(att4, 1.0);
        instance.setValue(priceAttribute, 135646923);
        trainingSet.add(instance);


        Instances dataset = new Instances("dataSet", attributes, featuresList.size());
        dataset.setClassIndex(dataset.numAttributes() - 1);

        for (int i = 0; i < featuresList.size(); i++) {
            instance = new DenseInstance(5);
            instance.setValue(att1, featuresList.get(i)[0]);
            instance.setValue(att2, featuresList.get(i)[1]);
            instance.setValue(att3, featuresList.get(i)[2]);
            instance.setValue(att4, featuresList.get(i)[3]);
            instance.setValue(priceAttribute, labels.get(i));
            dataset.add(instance);
        }

        trainingSet = dataset;
        Instances validationSet  = dataset;



        // create the target function and train it by calling the build methods
        Classifier targetFunction = new LinearRegression();
        targetFunction.buildClassifier(trainingSet);
        System.out.println("targetFunction " + targetFunction);

        // evaluate
        Evaluation evaluation = new Evaluation(trainingSet);
        evaluation.evaluateModel(targetFunction, validationSet);
        System.out.println(evaluation.toSummaryString("Results", false));


        // predict
        Instances unlabeledInstances = new Instances("trainData", attributes, 1);
        unlabeledInstances.setClassIndex(trainingSet.numAttributes() - 1);
        Instance unlabeled = new DenseInstance(4);
        unlabeled.setValue(att1, 500.0);
        unlabeled.setValue(att2, 5.0);
        unlabeled.setValue(att3, 4.0);
        unlabeled.setValue(att4, 3.0);
        unlabeledInstances.add(unlabeled);

        double prediction  = targetFunction.classifyInstance(unlabeledInstances.get(0));
        System.out.printf("Pred: %.01f\n", prediction);


//        Graph graph = Graph.create(Data.getFirstColumn(featuresList), labels, "house prices", "Price(€) in 1000´s", "Size in m²");
//        graph.addLine("weka", x -> {
//            try {
//                Instance inst = new DenseInstance(3);
//                inst.setValue(sizeAttribute, x);
//                inst.setValue(squaredSizeAttribute, Math.pow(x, 2));
//                return targetFunction.classifyInstance(inst);
//            } catch (Exception e) {
//                throw new RuntimeException(e);
//            }
//        });

//        graph.display();
    }
}

