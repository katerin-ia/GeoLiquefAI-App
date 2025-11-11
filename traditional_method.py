import numpy as np

def calculate_traditional_fs(input_data: dict):
    """
    Calcula el Factor de Seguridad (FS) usando el método tradicional
    simplificado (Seed & Idriss), basado en las fórmulas de tu notebook.
    """
    
    try:
        # Extraer variables del diccionario de entrada
        z = input_data.get("z_m")           # Profundidad (m)
        a_max = input_data.get("a_max")     # Aceleración máxima (g)
        sv = input_data.get("estres_v_total") # Esfuerzo total (kPa)
        sv_eff = input_data.get("estres_v_ef") # Esfuerzo efectivo (kPa)
        Mw = input_data.get("Mw")           # Magnitud
        N1_60_cs = input_data.get("N1_60_cs") # SPT corregido

        g = 9.81 # Aceleración de la gravedad
        Pa = 101.3  # Presión atmosférica en kPa

        # 1. Cálculo del Esfuerzo Cíclico (CSR)
        # Factor de reducción de esfuerzo (rd) - (Liao y Whitman, 1986)
        if z <= 9.15:
            rd = 1.0 - 0.00765 * z
        else:
            rd = 1.174 - 0.0267 * z
            
        CSR = 0.65 * (a_max) * (sv / sv_eff) * rd # a_max ya está en g

        # 2. Factor de Escala de Magnitud (MSF) - (Idriss & Boulanger 2014)
        MSF = 6.9 * np.exp(-Mw / 4) - 0.058
        if MSF < 0.69: MSF = 0.69 # Límite inferior
        
        # 3. Factor de Corrección por Sobrecarga (K_sigma)
        # (Boulanger & Idriss, 2014)
        f = 0.5 # Asumiendo arenas
        K_sigma = 1 - 0.007 * ((sv_eff / Pa)**1.32 - 1) # Simplificado
        K_sigma = min(K_sigma, 1.1) # Límite superior

        # 4. Resistencia Cíclica (CRR) - (Idriss & Boulanger 2014 para N1_60_cs)
        # Esta es una fórmula común y robusta
        if N1_60_cs < 37:
            CRR_7_5 = np.exp(
                (N1_60_cs / 14.1) + 
                (N1_60_cs / 126)**2 - 
                (N1_60_cs / 23.6)**3 + 
                (N1_60_cs / 25.4)**4 - 2.8
            )
        else:
            # Suelo demasiado denso para licuarse por este método
            CRR_7_5 = 2.0 # Un valor alto para indicar no licuefacción


        # 5. Factor de Seguridad (FS)
        FS_trad = (CRR_7_5 * MSF * K_sigma) / CSR
        
        return {
            "FS_trad": FS_trad,
            "CSR": CSR,
            "CRR_adj": CRR_7_5 * MSF * K_sigma,
            "rd": rd,
            "MSF": MSF,
            "K_sigma": K_sigma
        }

    except Exception as e:
        print(f"Error en cálculo tradicional: {e}")
        return {
            "FS_trad": None,
            "CSR": None,
            "CRR_adj": None
        }