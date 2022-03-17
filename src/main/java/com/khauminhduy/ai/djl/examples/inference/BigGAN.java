package com.khauminhduy.ai.djl.examples.inference;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;

public class BigGAN {

	private BigGAN() {
	}

	public static void saveImages(Image[] generatedImages) throws IOException {
		Path outputPath = Paths.get("build/output/gan/");
		Files.createDirectories(outputPath);

		for (int i = 0; i < generatedImages.length; i++) {
			Path imagePath = outputPath.resolve("/image" + i + ".png");
			generatedImages[i].save(Files.newOutputStream(imagePath), "png");
		}
		System.out.println("Generated images have been saved in: " + outputPath);
	}

	public static Image[] generate()
			throws ModelNotFoundException, MalformedModelException, IOException, TranslateException {
		// @formatter:off
		Criteria<int[], Image[]> criteria = 
				Criteria.builder()
						.optApplication(Application.CV.IMAGE_GENERATION)
						.setTypes(int[].class, Image[].class)
						.optFilter("size", "256")
						.optArgument("truncation", 0.4f)
						.optEngine("PyTorch")
						.optProgress(new ProgressBar())
						.build();
		// @formatter:on
		int[] input = { 100, 27, 971, 970, 933 };

		System.out.println(criteria);
		try (ZooModel<int[], Image[]> models = criteria.loadModel();) {
			try (Predictor<int[], Image[]> predictor = models.newPredictor();) {
				return predictor.predict(input);
			}
		}
	}

	public static void main(String[] args) {
		try {
			Image[] images = BigGAN.generate();
			System.out.println(String.format("Using PyTorch Engine. %d images generated.", images.length));
			saveImages(images);
		} catch (ModelNotFoundException | MalformedModelException | IOException | TranslateException e) {
			e.printStackTrace();
		}
	}

}
