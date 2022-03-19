package com.khauminhduy.ai.djl.examples.inference;

import java.io.IOException;
import java.util.List;
import java.util.Map;

import ai.djl.Application;
import ai.djl.repository.Artifact;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;

public class ListModels {

	private ListModels() {
	}

	public static void main(String[] args) {
		try {
			Map<Application, List<Artifact>> models = ModelZoo.listModels();
			models.forEach((app, list) -> {
				String appName = app.toString();
				list.forEach(model -> {
					System.out.println("AppName: " + appName + " || model: " + model);
				});
			});
		} catch (ModelNotFoundException | IOException e) {
			e.printStackTrace();
		}
	}

}
