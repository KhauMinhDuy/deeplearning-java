package com.khauminhduy.ai.djl.examples.inference;

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

import javax.imageio.ImageIO;

import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.SingleShotDetectionTranslator;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;

public class ObjectDetection {

	public static void predict()
			throws IOException, ModelNotFoundException, MalformedModelException, TranslateException {
//		Path imageFile = Paths.get("src/main/resources/d.jpg");
//		Image img = ImageFactory.getInstance().fromFile(imageFile);
		BufferedImage bufferedImage = ImageIO.read(new File("src/main/resources/c.png"));
		Image img = ImageFactory.getInstance().fromImage(bufferedImage);
		String backbone;
		if ("TensorFlow".equalsIgnoreCase(Engine.getDefaultEngineName())) {
			backbone = "mobilenet_v2";
		} else {
			backbone = "resnet50";
		}

		Translator<Image, DetectedObjects> translator = SingleShotDetectionTranslator.builder()
	            .addTransform(new ToTensor())
	            .optSynsetArtifactName("synset.txt")
//	            .optSynsetUrl("https://mysynset.txt")
	            .build();
		Criteria<Image, DetectedObjects> criteria = 
				Criteria.builder()
						.optApplication(Application.CV.OBJECT_DETECTION)
						.setTypes(Image.class, DetectedObjects.class)
//						.optModelPath(Paths.get(System.getProperty("user.home") + "/BHX-Pos/resnet50_ssd/resnet50_ssd_model-0000.params"))
//						.optModelName("BHX-Pos/resnet50_ssd")
//						.optTranslator(translator)
						.optFilter("backbone", backbone)
						.optEngine(Engine.getDefaultEngineName())
						.optProgress(new ProgressBar())
						.build();
		try (ZooModel<Image, DetectedObjects> model = criteria.loadModel();) {
			try (Predictor<Image, DetectedObjects> predictor = model.newPredictor();) {
				DetectedObjects detection = predictor.predict(img);
				saveBoundBoxImage(img, detection);
			}
		}
	}

	private static void saveBoundBoxImage(Image img, DetectedObjects detection) throws IOException {
		Path outputDir = Paths.get("build/output");
		Files.createDirectories(outputDir);

		List<DetectedObjects.DetectedObject> items = detection.items();
		int i = 1;
//		List<DetectedObjects.DetectedObject> items = detection.topK(3);
		for (DetectedObjects.DetectedObject item : items) {
			if ("person".equalsIgnoreCase(item.getClassName())) {
				Rectangle rect = item.getBoundingBox().getBounds();
				int width = img.getWidth();
				int height = img.getHeight();
				int x = (int) (rect.getX() * width);
				int y = (int) (rect.getY() * height);
				int widthImg = (int) (rect.getWidth() * width);
				if ((widthImg + x) > width) {
					widthImg = width - x;
				}
				int heightImg = (int) (rect.getHeight() * height);
				if ((heightImg + y) > height) {
					heightImg = height - y;
				}
				Image subImage = img.getSubImage(x, y, widthImg, heightImg);
				
				try (ByteArrayOutputStream baos = new ByteArrayOutputStream();) {
					String imagePath2 = outputDir.resolve(i++ + "_output.jpg").toString();
					subImage.save(baos, "jpg");
					BufferedImage read = ImageIO.read(new ByteArrayInputStream(baos.toByteArray()));
					ImageIO.write(read, "jpg", new File(imagePath2));
				}
			}
		}

		img.drawBoundingBoxes(detection);
		Path imagePath = outputDir.resolve("detected-dog_bike_car.png");
		img.save(Files.newOutputStream(imagePath), "png");
	}

	public static void main(String[] args) {
		try {
			ObjectDetection.predict();
			System.out.println("done");
		} catch (ModelNotFoundException | MalformedModelException | IOException | TranslateException e) {
			e.printStackTrace();
		}
	}

}
