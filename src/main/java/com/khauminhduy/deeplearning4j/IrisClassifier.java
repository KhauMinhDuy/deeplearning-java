package com.khauminhduy.deeplearning4j;

import java.io.IOException;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class IrisClassifier {

	private static final int CLASSES_COUNT = 3;
	private static final int FEATURES_COUNT = 4;

	public static void main(String[] args) throws IOException, InterruptedException {
		try (RecordReader recordReader = new CSVRecordReader(0, ',');) {
			recordReader.initialize(new FileSplit(new ClassPathResource("iris.txt").getFile()));
			DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, 150, FEATURES_COUNT,
					CLASSES_COUNT);
			DataSet dataSet = iterator.next();
			dataSet.shuffle(42);
			
			DataNormalization normalization = new NormalizerStandardize();
			normalization.fit(dataSet);
			normalization.transform(dataSet);
			
			SplitTestAndTrain testAndTrain = dataSet.splitTestAndTrain(0.65);
			DataSet trainingData = testAndTrain.getTrain();
			DataSet testData = testAndTrain.getTest();
			
			MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
							.iterations(1000)
							.activation(Activation.TANH)
							.weightInit(WeightInit.XAVIER)
							.learningRate(0.1)
							.regularization(true)
							.l2(0.0001)
							.list()
							.layer(0, new DenseLayer.Builder().nIn(FEATURES_COUNT).nOut(3).build())
							.layer(1, new DenseLayer.Builder().nIn(3).nOut(3).build())
							.layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
									.activation(Activation.SOFTMAX).nIn(3).nOut(CLASSES_COUNT).build())
							.backprop(true)
							.pretrain(false)
							.build();
			
		}
	}

}
