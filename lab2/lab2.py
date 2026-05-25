"""
Лабораторна робота №2
Тема: Базові моделі машинного навчання для зображень
Теми 4 + 5: MNIST класифікатор + CNN для CIFAR-10 + оцінка якості
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# ============================================================
# СЕКЦІЯ 1: Тренування простого класифікатора зображень (MNIST)
# ============================================================

def section1_mnist():
    print("=" * 60)
    print("СЕКЦІЯ 1: Класифікація MNIST")
    print("=" * 60)

    from tensorflow.keras.datasets import mnist

    # 1.1 Завантаження та ознайомлення з даними
    (x_train_full, y_train_full), (x_test_raw, y_test_raw) = mnist.load_data()
    print(f"Розмір тренувального набору: {x_train_full.shape}")
    print(f"Розмір тестового набору:     {x_test_raw.shape}")
    print(f"Тип даних: {x_train_full.dtype}, діапазон: [{x_train_full.min()}, {x_train_full.max()}]")
    print(f"Класи: {np.unique(y_train_full)}")

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle("1.1 — Приклади зображень MNIST (28×28, відтінки сірого)", fontsize=12)
    idx = np.random.choice(len(x_train_full), 10, replace=False)
    for i, ax in enumerate(axes.flat):
        ax.imshow(x_train_full[idx[i]], cmap='gray')
        ax.set_title(f"Клас: {y_train_full[idx[i]]}")
        ax.axis('off')
    plt.tight_layout()
    plt.savefig("s1_mnist_samples.png", dpi=120, bbox_inches='tight')
    plt.show()

    # 1.2 Препроцесинг
    from sklearn.model_selection import train_test_split

    x_full = np.concatenate([x_train_full, x_test_raw], axis=0)
    y_full = np.concatenate([y_train_full, y_test_raw], axis=0)

    x_flat = x_full.reshape((x_full.shape[0], -1))
    x_norm = x_flat.astype("float32") / 255.0

    x_train, x_test, y_train, y_test = train_test_split(
        x_norm, y_full, test_size=0.2, random_state=42
    )
    print(f"\n1.2 Препроцесинг:")
    print(f"  Вектор ознак: {x_train.shape[1]} (28x28=784)")
    print(f"  Train: {x_train.shape[0]}, Test: {x_test.shape[0]}")
    print(f"  Діапазон після нормалізації: [{x_norm.min():.2f}, {x_norm.max():.2f}]")

    results = {}

    # 1.3.1 Логістична регресія
    print("\n1.3.1 Логістична регресія...")
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from time import time

    scaler = StandardScaler()
    x_tr_sc = scaler.fit_transform(x_train)
    x_te_sc = scaler.transform(x_test)

    clf = LogisticRegression(solver="saga", multi_class="multinomial",
                             max_iter=200, n_jobs=-1, verbose=0)
    t0 = time()
    clf.fit(x_tr_sc, y_train)
    train_t = time() - t0

    t0 = time()
    y_pred_lr = clf.predict(x_te_sc)
    pred_t = time() - t0

    acc_lr = np.mean(y_pred_lr == y_test)
    results["LogReg"] = {"acc": acc_lr, "train_t": train_t, "pred_t": pred_t}
    print(f"  Точність: {acc_lr:.4f} ({acc_lr*100:.2f}%)")
    print(f"  Час навчання: {train_t:.1f}с, передбачення: {pred_t:.3f}с")

    # 1.3.2 Одношаровий MLP (Keras)
    print("\n1.3.2 MLP (Keras)...")
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    mlp_model = keras.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])
    mlp_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    t0 = time()
    history_mlp = mlp_model.fit(
        x_train, y_train,
        epochs=10, batch_size=128,
        validation_split=0.2,
        verbose=0
    )
    train_t_mlp = time() - t0

    t0 = time()
    test_loss, test_acc_mlp = mlp_model.evaluate(x_test, y_test, verbose=0)
    pred_t_mlp = time() - t0

    results["MLP"] = {"acc": test_acc_mlp, "train_t": train_t_mlp, "pred_t": pred_t_mlp}
    print(f"  Точність: {test_acc_mlp:.4f} ({test_acc_mlp*100:.2f}%)")
    print(f"  Час навчання: {train_t_mlp:.1f}с, передбачення: {pred_t_mlp:.3f}с")

    # Графік loss MLP
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("1.3.2 — MLP навчання на MNIST", fontsize=12)
    axes[0].plot(history_mlp.history['loss'], label='train loss')
    axes[0].plot(history_mlp.history['val_loss'], label='val loss')
    axes[0].set_title("Функція втрат")
    axes[0].set_xlabel("Епоха")
    axes[0].legend()

    axes[1].plot(history_mlp.history['accuracy'], label='train acc')
    axes[1].plot(history_mlp.history['val_accuracy'], label='val acc')
    axes[1].set_title("Точність")
    axes[1].set_xlabel("Епоха")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig("s1_mlp_training.png", dpi=120, bbox_inches='tight')
    plt.show()

    # 1.4 Візуалізація передбачень MLP
    probs = mlp_model.predict(x_test, batch_size=256, verbose=0)
    y_pred_mlp = probs.argmax(axis=1)
    correct_idx   = np.where(y_pred_mlp == y_test)[0]
    incorrect_idx = np.where(y_pred_mlp != y_test)[0]
    uncertain_idx = np.argsort(probs.max(axis=1))[:10]  # найнижча впевненість

    def show_predictions(indices, title, n=10):
        fig, axes = plt.subplots(2, 5, figsize=(12, 5))
        fig.suptitle(title, fontsize=12)
        for i, ax in enumerate(axes.flat):
            if i >= len(indices): ax.axis('off'); continue
            idx = indices[i]
            img = x_test[idx].reshape(28, 28)
            ax.imshow(img, cmap='gray')
            conf = probs[idx].max()
            ax.set_title(f"Прав:{y_test[idx]} Пред:{y_pred_mlp[idx]}\n{conf:.2f}", fontsize=8)
            ax.axis('off')
        plt.tight_layout()
        return fig

    fig1 = show_predictions(correct_idx[:10], "1.4 — Правильні передбачення MLP")
    fig1.savefig("s1_correct.png", dpi=120, bbox_inches='tight')
    plt.show()

    fig2 = show_predictions(incorrect_idx[:10], "1.4 — Неправильні передбачення MLP")
    fig2.savefig("s1_incorrect.png", dpi=120, bbox_inches='tight')
    plt.show()

    fig3 = show_predictions(uncertain_idx, "1.4 — Невпевнені передбачення MLP (найнижча довіра)")
    fig3.savefig("s1_uncertain.png", dpi=120, bbox_inches='tight')
    plt.show()

    # 1.5 Порівняння KNN та SVM
    print("\n1.5 KNN та LinearSVC (на підвибірці 10 000)...")
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import LinearSVC
    from sklearn.pipeline import Pipeline

    n_sub = 10000
    rng = np.random.default_rng(42)
    sub_idx = rng.choice(len(x_train), n_sub, replace=False)
    x_sub, y_sub = x_train[sub_idx], y_train[sub_idx]

    models_cmp = {
        "KNN(k=3)": Pipeline([("scaler", StandardScaler()),
                               ("clf", KNeighborsClassifier(n_neighbors=3))]),
        "SVM(Linear)": Pipeline([("scaler", StandardScaler()),
                                  ("clf", LinearSVC(C=2.0, random_state=42, max_iter=5000))])
    }

    for name, pipe in models_cmp.items():
        t0 = time()
        pipe.fit(x_sub, y_sub)
        tr_t = time() - t0
        t0 = time()
        y_p = pipe.predict(x_test)
        pr_t = time() - t0
        acc = np.mean(y_p == y_test)
        results[name] = {"acc": acc, "train_t": tr_t, "pred_t": pr_t}
        print(f"  {name}: точність={acc:.4f}, навчання={tr_t:.1f}с, передбачення={pr_t:.3f}с")

    # Порівняльний графік
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("1.5 — Порівняння моделей на MNIST", fontsize=12)
    names = list(results.keys())
    accs  = [results[n]["acc"] for n in names]
    tr_ts = [results[n]["train_t"] for n in names]
    pr_ts = [results[n]["pred_t"] for n in names]

    axes[0].bar(names, [a*100 for a in accs], color=['#4C72B0','#DD8452','#55A868','#C44E52'])
    axes[0].set_title("Точність (%)"); axes[0].set_ylim([85, 100])
    for i, v in enumerate(accs): axes[0].text(i, v*100+0.1, f"{v*100:.2f}%", ha='center', fontsize=9)

    axes[1].bar(names, tr_ts, color=['#4C72B0','#DD8452','#55A868','#C44E52'])
    axes[1].set_title("Час навчання (с)")

    axes[2].bar(names, pr_ts, color=['#4C72B0','#DD8452','#55A868','#C44E52'])
    axes[2].set_title("Час передбачення (с)")

    plt.tight_layout()
    plt.savefig("s1_model_comparison.png", dpi=120, bbox_inches='tight')
    plt.show()

    print("\n--- Підсумок Секції 1 ---")
    for name, r in results.items():
        print(f"  {name:12s}: acc={r['acc']:.4f}  train={r['train_t']:.1f}s  pred={r['pred_t']:.3f}s")

    return mlp_model, x_test, y_test


# ============================================================
# СЕКЦІЯ 2: CNN для CIFAR-10
# ============================================================

def section2_cifar10():
    print("\n" + "=" * 60)
    print("СЕКЦІЯ 2: CNN для CIFAR-10")
    print("=" * 60)

    from tensorflow.keras.datasets import cifar10
    from tensorflow.keras.utils import to_categorical
    from tensorflow import keras
    from tensorflow.keras import layers

    CLASS_NAMES = ["airplane","automobile","bird","cat","deer",
                   "dog","frog","horse","ship","truck"]

    # 2.1 Завантаження та підготовка даних
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print(f"Train: {x_train.shape}, Test: {x_test.shape}")
    print(f"Тип: {x_train.dtype}, діапазон: [{x_train.min()}, {x_train.max()}]")

    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32") / 255.0
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat  = to_categorical(y_test, 10)

    # Приклади зображень по класах
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    fig.suptitle("2.1 — Приклади CIFAR-10 по класах (32×32, RGB)", fontsize=12)
    for i, ax in enumerate(axes.flat):
        idx = np.where(y_train.flatten() == i)[0][0]
        ax.imshow(x_train[idx])
        ax.set_title(CLASS_NAMES[i])
        ax.axis('off')
    plt.tight_layout()
    plt.savefig("s2_cifar_samples.png", dpi=120, bbox_inches='tight')
    plt.show()

    # 2.2 Архітектура CNN
    num_classes  = 10
    input_shape  = (32, 32, 3)

    model = keras.Sequential([
        layers.Conv2D(32, (3,3), padding="same", activation="relu", input_shape=input_shape),
        layers.Conv2D(32, (3,3), activation="relu"),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3,3), padding="same", activation="relu"),
        layers.Conv2D(64, (3,3), activation="relu"),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\n2.2 Архітектура CNN:")
    model.summary()

    # 2.3 Навчання
    print("\n2.3 Навчання CNN (10 епох)...")
    from time import time
    t0 = time()
    history = model.fit(
        x_train, y_train_cat,
        epochs=10,
        batch_size=64,
        validation_split=0.2,
        verbose=1
    )
    train_time = time() - t0
    print(f"Час навчання: {train_time:.1f}с")

    test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
    train_loss, train_acc = model.evaluate(x_train, y_train_cat, verbose=0)
    print(f"Train: loss={train_loss:.4f}, acc={train_acc:.4f}")
    print(f"Test:  loss={test_loss:.4f},  acc={test_acc:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("2.3 — Навчання CNN на CIFAR-10", fontsize=12)
    axes[0].plot(history.history['loss'],     label='train')
    axes[0].plot(history.history['val_loss'], label='val')
    axes[0].set_title("Функція втрат"); axes[0].set_xlabel("Епоха"); axes[0].legend()
    axes[1].plot(history.history['accuracy'],     label='train')
    axes[1].plot(history.history['val_accuracy'], label='val')
    axes[1].set_title("Точність"); axes[1].set_xlabel("Епоха"); axes[1].legend()
    plt.tight_layout()
    plt.savefig("s2_cnn_training.png", dpi=120, bbox_inches='tight')
    plt.show()

    # 2.4 Збереження моделі та результатів
    model.save("cnn_cifar10_model.keras")
    print("Модель збережено: cnn_cifar10_model.keras")

    import csv
    with open("cnn_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Loss", "Accuracy"])
        writer.writerow(["Train", f"{train_loss:.4f}", f"{train_acc:.4f}"])
        writer.writerow(["Test",  f"{test_loss:.4f}",  f"{test_acc:.4f}"])
    print("Результати збережено: cnn_results.csv")

    return model, x_test, y_test.flatten(), y_test_cat, CLASS_NAMES


# ============================================================
# СЕКЦІЯ 3: Оцінка якості моделі
# ============================================================

def section3_evaluation(model, x_test, y_true, y_test_cat, class_names):
    print("\n" + "=" * 60)
    print("СЕКЦІЯ 3: Оцінка якості моделі CNN (CIFAR-10)")
    print("=" * 60)

    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns

    probs  = model.predict(x_test, batch_size=256, verbose=0)
    y_pred = probs.argmax(axis=1)

    # 3.1 Метрики
    print("\n3.1 Класифікаційний звіт:")
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)

    with open("classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    # 3.2 Матриця невідповідностей
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Передбачений клас")
    ax.set_ylabel("Справжній клас")
    ax.set_title("3.2 — Матриця невідповідностей (CIFAR-10)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("s3_confusion_matrix.png", dpi=120, bbox_inches='tight')
    plt.show()

    # 3.3 Візуалізація помилок по класах (10 помилок на клас)
    print("\n3.3 Помилки по класах:")
    fig, axes = plt.subplots(10, 10, figsize=(16, 16))
    fig.suptitle("3.3 — По 10 помилкових прикладів для кожного класу", fontsize=12)

    for cls_i in range(10):
        cls_errors = np.where((y_true == cls_i) & (y_pred != cls_i))[0]
        for j in range(10):
            ax = axes[cls_i, j]
            if j < len(cls_errors):
                idx = cls_errors[j]
                ax.imshow(x_test[idx])
                pred_lbl = class_names[y_pred[idx]]
                conf = probs[idx].max()
                ax.set_title(f"→{pred_lbl[:4]}\n{conf:.2f}", fontsize=6)
            else:
                ax.axis('off')
                continue
            ax.axis('off')
        axes[cls_i, 0].set_ylabel(class_names[cls_i], fontsize=8, rotation=0,
                                   ha='right', va='center')

    plt.tight_layout()
    plt.savefig("s3_error_examples.png", dpi=120, bbox_inches='tight')
    plt.show()

    # Топ-3 найгірших класів
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    print("\nТочність по класах:")
    for i, (name, acc) in enumerate(zip(class_names, per_class_acc)):
        print(f"  {name:12s}: {acc:.4f} ({acc*100:.1f}%)")

    worst = np.argsort(per_class_acc)[:3]
    print(f"\nНайгірші класи: {[class_names[w] for w in worst]}")
    print("(Типово: cat/dog/bird важко розрізнити через схожий вигляд)")


# ============================================================
# ГОЛОВНА ФУНКЦІЯ
# ============================================================

if __name__ == "__main__":
    print("Лабораторна робота №2 — Базові моделі ML для зображень\n")

    # Секція 1: MNIST
    mlp_model, x_test_mnist, y_test_mnist = section1_mnist()

    # Секція 2: CIFAR-10 CNN
    cnn_model, x_test_cifar, y_test_cifar, y_test_cat, class_names = section2_cifar10()

    # Секція 3: Оцінка якості CNN
    section3_evaluation(cnn_model, x_test_cifar, y_test_cifar, y_test_cat, class_names)

    print("\n" + "=" * 60)
    print("Всі секції виконано!")
    print("Збережені файли:")
    files = [
        "s1_mnist_samples.png", "s1_mlp_training.png",
        "s1_correct.png", "s1_incorrect.png", "s1_uncertain.png",
        "s1_model_comparison.png",
        "s2_cifar_samples.png", "s2_cnn_training.png",
        "cnn_cifar10_model.keras", "cnn_results.csv",
        "s3_confusion_matrix.png", "s3_error_examples.png",
        "classification_report.txt",
    ]
    for f in files:
        print(f"  - {f}")
