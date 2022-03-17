package com.khauminhduy.ai.djl.examples.inference;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;

public class ActionRecognition {

	private ActionRecognition() {
	}

	public static Classifications predict()
			throws IOException, ModelNotFoundException, MalformedModelException, TranslateException {
		Path imageFile = Paths.get("src/main/resources/action_discus_throw.png");
//		Path imageFile = Paths.get("src/main/resources/a.jpg");
		Image img = ImageFactory.getInstance().fromFile(imageFile);

		// @formatter:off
		Criteria<Image, Classifications> criteria = 
						Criteria.builder()
							.optApplication(Application.CV.ACTION_RECOGNITION)
							.setTypes(Image.class, Classifications.class)
							.optFilter("backbone", "inceptionv3")
							.optFilter("dataset", "ucf101")
							.optEngine("MXNet").optProgress(new ProgressBar())
							.build();
		// @formatter:on

		try (ZooModel<Image, Classifications> inception = criteria.loadModel();) {
			try (Predictor<Image, Classifications> predictor = inception.newPredictor();) {
				return predictor.predict(img);
			}
		}
	}

	public static void main(String[] args) {
		try {
			Classifications classifications = ActionRecognition.predict();
			System.out.println(classifications);
		} catch (ModelNotFoundException | MalformedModelException | IOException | TranslateException e) {
			e.printStackTrace();
		}
	}

}
