package machinelearning;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

public class Example1Lineal {

    public static void main(String[] args) throws IOException {

        // traer las etiquetas, y la informacion de caracteristicas
        List<Double> labels = ZData.loadLabels("/example1.txt");
        List<Double[]> datasetFile = ZData.loadFeaturesList("/example1.txt");
        List<Double[]> dataset = datasetFile.stream().map(features -> new Double[]{1.0, features[0]}).collect(Collectors.toList());

        // crear la función escalable
        Function<Double[], Double[]> scalingFunc = FeaturesScaling.createFunction(dataset);
        List<Double[]> scaledDataset = dataset.stream().map(scalingFunc).collect(Collectors.toList());

        //X crear funcion de hipotesis y entrenar a la funcion para minimizar el costo
        LinearRegressionFunction targetFunction = new LinearRegressionFunction(new double[]{1.0, 1.0});

        for (int i = 0; i < 1000; i++) { // cantidad de iteraciones
            targetFunction = Learner.train(targetFunction, scaledDataset, labels, 0.1);
        }

        //X hacer una prediccion para una casa de 190 metros
        Double[] scaledFeatureVector = scalingFunc.apply(new Double[]{1.0, 190.0});
        double predictedPrice = targetFunction.apply(scaledFeatureVector);
        System.out.printf("Predicción: $%.01f\n", predictedPrice);

        // hacer el gráfico
        Graph graph = Graph.create(ZData.getFirstColumn(datasetFile), labels, "Precios", "Precio en USD", "Tamaño");
        final LinearRegressionFunction func = targetFunction;
        graph.addLine("plain", x -> func.apply(scalingFunc.apply(new Double[]{1.0, x})));
        graph.display();
    }
}

//        double[] consultas = {190.0, 200.0};        
//        for (int i = 0; i < consultas.length; i++) {
//            Double[] scaledFeatureVector = scalingFunc.apply(new Double[]{1.0, consultas[i]});
//            double predictedPrice = targetFunction.apply(scaledFeatureVector);
//            System.out.printf("Predicción para "+consultas[i]+": $%.01f\n", predictedPrice);
//        }


//Map<Integer, Integer> costHistory = new HashMap<>();


            //Double cost = Cost.cost(targetFunction, scaledDataset, labels);
            //costHistory.put(i + 1, (int) Math.round(cost));