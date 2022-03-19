package com.khauminhduy.ai.djl.examples.inference;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Scanner;
import java.util.concurrent.ConcurrentHashMap;

import com.google.gson.annotations.SerializedName;

import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.TranslateException;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.JsonUtils;

public class ObjectDetectionWithTensorflowSavedModel {

	private ObjectDetectionWithTensorflowSavedModel() {
	}

	public static DetectedObjects predict() throws IOException, ModelNotFoundException, MalformedModelException, TranslateException {
		Path imagePath = Paths.get("src/main/resources/dog-cat.jpg");
		Image img = ImageFactory.getInstance().fromFile(imagePath);
//		String modelUrl = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz";
		String modelUrl = "C:\\Users\\duykh\\Downloads\\ssd_mobilenet_v2_320x320_coco17_tpu-8.tar\\ssd_mobilenet_v2_320x320_coco17_tpu-8";
		// @formatter:off
		Criteria<Image, DetectedObjects> criteria =
					Criteria.builder()
							.optApplication(Application.CV.OBJECT_DETECTION)
							.setTypes(Image.class, DetectedObjects.class)
//							.optModelUrls(modelUrl)
							.optModelPath(Paths.get(modelUrl))
							.optModelName("ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model")
							.optTranslator(new MyTranslator())
							.optEngine(Engine.getDefaultEngineName())
							.optProgress(new ProgressBar())
							.build();
		// @formatter:on
		try(ZooModel<Image, DetectedObjects> models = criteria.loadModel();
				Predictor<Image,DetectedObjects> predictor = models.newPredictor();) {
			DetectedObjects detectedObjects = predictor.predict(img);
			saveBoundingBoxImage(img, detectedObjects);
			return detectedObjects;
		}

	}
	
	

	public static Map<Integer, String> loadSysset() throws IOException {
		URL synsetUrl = new URL(
				"https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/mscoco_label_map.pbtxt");
		Map<Integer, String> map = new ConcurrentHashMap<>();
		int maxId = 0;
		try (InputStream is = new BufferedInputStream(synsetUrl.openStream());
				Scanner scanner = new Scanner(is, StandardCharsets.UTF_8.name());) {
			scanner.useDelimiter("item ");
			while (scanner.hasNext()) {
				String content = scanner.next();
				content = content.replaceAll("(\"|\\d)\\n\\s", "$1,");
				Item item = JsonUtils.GSON.fromJson(content, Item.class);
				map.put(item.id, item.displayName);
				if (item.id > maxId) {
					maxId = item.id;
				}
			}
		}
		return map;
	}
	
	private static void saveBoundingBoxImage(Image img, DetectedObjects detection) throws IOException {
		Path outputDir = Paths.get("build/output");
		Files.createDirectories(outputDir);
		img.drawBoundingBoxes(detection);
		Path imagePath = outputDir.resolve("detected-tensorflow-model-dog_bike_car.png");
		img.save(Files.newOutputStream(imagePath), "png");
	}

	private static final class Item {
		int id;

		@SerializedName("display_name")
		String displayName;
	}

	private static final class MyTranslator implements NoBatchifyTranslator<Image, DetectedObjects> {

		private Map<Integer, String> classes;
		private int maxBoxes;
		private float threshold;

		public MyTranslator() {
			maxBoxes = 10;
			threshold = 0.7f;
		}

		@Override
		public NDList processInput(TranslatorContext ctx, Image input) throws Exception {
			NDArray array = input.toNDArray(ctx.getNDManager(), Image.Flag.COLOR);
			array = NDImageUtils.resize(array, 224);
			array = array.toType(DataType.UINT8, true);
			array = array.expandDims(0);
			return new NDList(array);
		}

		@Override
		public void prepare(TranslatorContext ctx) throws Exception {
			if (classes == null) {
				classes = loadSysset();
			}
		}

		@Override
		public DetectedObjects processOutput(TranslatorContext ctx, NDList list) throws Exception {
			int[] classIds = null;
			float[] probabilities = null;
			NDArray boundingBoxes = null;
			for (NDArray array : list) {
				if ("detection_boxes".equalsIgnoreCase(array.getName())) {
					boundingBoxes = array.get(0);
				} else if ("detection_scores".equalsIgnoreCase(array.getName())) {
					probabilities = array.get(0).toFloatArray();
				} else if ("detection_classes".equalsIgnoreCase(array.getName())) {
					classIds = array.get(0).toType(DataType.INT32, true).toIntArray();
				}
			}
			Objects.requireNonNull(classIds);
			Objects.requireNonNull(probabilities);
			Objects.requireNonNull(boundingBoxes);

			List<String> retNames = new ArrayList<>();
			List<Double> retProbs = new ArrayList<>();
			List<BoundingBox> retBB = new ArrayList<>();

			for (int i = 0; i < Math.min(classIds.length, maxBoxes); i++) {
				int classId = classIds[i];
				double probability = probabilities[i];
				if (classId > 0 && probability > threshold) {
					String className = classes.getOrDefault(classId, "#" + classId);
					float[] box = boundingBoxes.get(i).toFloatArray();
					float yMin = box[0];
					float xMin = box[1];
					float yMax = box[2];
					float xMax = box[3];
					Rectangle rect = new Rectangle(xMin, yMin, xMax - xMin, yMax - yMin);
					retNames.add(className);
					retProbs.add(probability);
					retBB.add(rect);
				}
			}

			return new DetectedObjects(retNames, retProbs, retBB);
		}

	}
	
	public static void main(String[] args) {
		try {
			DetectedObjects detectedObjects = ObjectDetectionWithTensorflowSavedModel.predict();
			System.out.println(detectedObjects);
		} catch (ModelNotFoundException | MalformedModelException | IOException | TranslateException e) {
			e.printStackTrace();
		}
	}

}
