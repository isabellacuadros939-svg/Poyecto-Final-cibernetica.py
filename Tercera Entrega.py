import numpy as np
import matplotlib.pyplot as plt
from control_agente import simular_sistema


def calcular_metricas(x, u, ref=0.0):
    mae = float(np.mean(np.abs(x[1:] - ref)))
    error_final = float(np.abs(x[-1] - ref))
    uso_reserva = float(np.mean(u))
    sobreimpulso = float(np.max(np.maximum(0.0, x[1:] - ref)))
    return mae, error_final, uso_reserva, sobreimpulso


def ejecutar_experimentos_entrega3():
    pasos = 80
    semilla = 42

    d_nominal = np.zeros(pasos)
    d_mod = np.zeros(pasos)
    d_mod[20:50] = -0.04

    d_sev = np.zeros(pasos)
    d_sev[20:50] = -0.15

    escenarios = [
        ("A. Nominal", d_nominal, 0.0, 0.0),
        ("B. Perturbacion Moderada", d_mod, 0.0, 0.0),
        ("C. Perturbacion Severa", d_sev, 0.0, 0.0),
        ("D. Ruido e Info Incompleta", d_mod, 0.06, 0.15),
    ]

    print(f"{'Escenario':<28} | {'Estrategia':<12} | {'MAE':<8} | {'ErrFinal':<8} | {'Reserva(%)':<10} | {'Sobreimp.':<9}")
    print("-" * 90)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for idx, (nombre, perfil_d, ruido, perdida) in enumerate(escenarios):
        x_b, u_b, _ = simular_sistema(
            modo="BASE", pasos=pasos, usar_agente=False,
            perfil_d=perfil_d, ruido_std=ruido, prob_perdida=perdida, semilla=semilla
        )
        x_m, u_m, _ = simular_sistema(
            modo="PI", pasos=pasos, usar_agente=True,
            perfil_d=perfil_d, ruido_std=ruido, prob_perdida=perdida, semilla=semilla
        )

        mae_b, ef_b, res_b, sob_b = calcular_metricas(x_b, u_b)
        mae_m, ef_m, res_m, sob_m = calcular_metricas(x_m, u_m)

        print(f"{nombre:<28} | {'Linea Base':<12} | {mae_b:<8.4f} | {ef_b:<8.4f} | {res_b:<10.2f} | {sob_b:<9.4f}")
        print(f"{nombre:<28} | {'PI + Agente':<12} | {mae_m:<8.4f} | {ef_m:<8.4f} | {res_m:<10.2f} | {sob_m:<9.4f}")
        print("-" * 90)

        ax = axes[idx]
        ax.plot(x_b, label="Línea Base", linestyle="--")
        ax.plot(x_m, label="PI + Agente")
        ax.axhline(0.0, linestyle=":", color="black", label="Referencia")
        ax.set_title(nombre)
        ax.set_xlabel("Paso k")
        ax.set_ylabel("Desviación x[k]")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.savefig("escenarios_entrega3_comparacion.png", dpi=150)
    plt.close()

    print("\n--- PRUEBA DE SENSIBILIDAD (Variando Ki en Perturbación Moderada) ---")
    for ki_val in [1.0, 6.0, 15.0, 28.0]:
        x_s, u_s, _ = simular_sistema(
            modo="PI", pasos=pasos, ki=ki_val, usar_agente=True,
            perfil_d=d_mod, semilla=semilla
        )
        mae_s, ef_s, res_s, sob_s = calcular_metricas(x_s, u_s)
        print(f"Ki = {ki_val:<5.1f} | MAE: {mae_s:.4f} | ErrFinal: {ef_s:.4f} | ReservaProm: {res_s:.2f}% | Sobreimpulso: {sob_s:.4f}")


if __name__ == "__main__":
    ejecutar_experimentos_entrega3()