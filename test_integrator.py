if __name__ == "__main__":
    print("✅ Prueba de imports iniciada...")
    import numpy as np
    from src.modulos.load_model import model_fun
    from src.modulos.preprocess_img import preprocess
    from src.modulos.grad_cam import grad_cam

    # Verifica que NumPy esté disponible
    print("NumPy versión:", np.__version__)

    # Verifica que las funciones importadas respondan
    try:
        print("🔍 Probando model_fun()...")
        model = model_fun()
        print("✅ model_fun cargado correctamente:", type(model))

        print("🔍 Probando preprocess() con imagen dummy...")
        dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
        processed = preprocess(dummy_img)
        print("✅ preprocess ejecutado:", type(processed))

        print("🔍 Probando grad_cam() con entradas dummy...")
        cam_result = grad_cam(model, dummy_img)
        print("✅ grad_cam ejecutado:", type(cam_result))

    except Exception as e:
        print("❌ Error durante la prueba de funciones:", e)

    print("🎉 Prueba de imports completada.")