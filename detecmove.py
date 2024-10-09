import cv2
import numpy as np
import time
import threading
import pyaudio
import wave
import os
from moviepy.editor import VideoFileClip, AudioFileClip
from playsound import playsound

# Evento para controlar la reproducción del sonido
sound_playing = threading.Event()

def play_alert_sound():
    """Reproduce un sonido de alerta si no se está reproduciendo ya."""
    if not sound_playing.is_set():
        sound_playing.set()  # Marca el evento como activo
        playsound('alerta.mp3')
        sound_playing.clear()  # Desmarca el evento

def record_audio(filename, duration):
    """Graba audio del micrófono durante un tiempo especificado."""
    chunk = 1024  # Tamaño del bloque de datos
    format = pyaudio.paInt16  # Formato de audio
    channels = 1  # Mono
    rate = 44100  # Frecuencia de muestreo

    p = pyaudio.PyAudio()

    # Abre el flujo de audio
    stream = p.open(format=format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)

    # Captura los datos de audio
    frames = [stream.read(chunk) for _ in range(int(rate / chunk * duration))]

    # Detiene y cierra el flujo de audio
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Guarda los datos de audio en un archivo WAV
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))

def combine_video_audio(video_filename, audio_filename, output_filename):
    """Combina un archivo de video y un archivo de audio en un solo archivo de salida."""
    try:
        # Carga el video y el audio
        video_clip = VideoFileClip(video_filename)
        audio_clip = AudioFileClip(audio_filename)

        # Asigna el audio al video
        final_clip = video_clip.set_audio(audio_clip)

        # Escribe el archivo de salida
        final_clip.write_videofile(output_filename, codec='libx264', audio_codec='aac')
        print(f"Archivo combinado guardado como {output_filename}")
    except Exception as e:
        print("Error al combinar video y audio:", e)

def main():
    """Función principal para la detección de movimiento y grabación."""
    # Crear carpetas si no existen
    os.makedirs('videos', exist_ok=True)
    os.makedirs('audios', exist_ok=True)
    os.makedirs('resultado_final', exist_ok=True)

    cap = cv2.VideoCapture(0)  # Inicializa la captura de video desde la cámara
    ret, frame1 = cap.read()  # Lee el primer frame
    ret, frame2 = cap.read()  # Lee el segundo frame

    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Configuración del códec de video
    out = None
    recording = False

    while cap.isOpened():
        # Calcula la diferencia entre los frames
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Dibuja contornos si se detecta movimiento
        motion_detected = False
        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue
            motion_detected = True
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame1, "Peligro Detectado!", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Reproduce el sonido de alerta en un hilo separado
            if not sound_playing.is_set():
                threading.Thread(target=play_alert_sound).start()

            # Inicia la grabación si no está grabando
            if not recording:
                timestamp = int(time.time())
                video_filename = f'videos/grabacion_{timestamp}.avi'
                audio_filename = f'audios/audio_{timestamp}.wav'
                output_filename = f'resultado_final/output_{timestamp}.mp4'
                out = cv2.VideoWriter(video_filename, fourcc, 20.0, (frame1.shape[1], frame1.shape[0]))
                recording = True
                start_time = time.time()

                # Inicia la grabación de audio en un hilo separado
                threading.Thread(target=record_audio, args=(audio_filename, 15)).start()

        # Graba el video si está en modo de grabación
        if recording:
            out.write(frame1)
            # Detiene la grabación después de 15 segundos
            if time.time() - start_time > 15:
                recording = False
                out.release()
                # Muestra el mensaje de conversión
                cv2.putText(frame1, "Convirtiendo...", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.imshow("Camara", frame1)
                cv2.waitKey(1)  # Actualiza la ventana para mostrar el mensaje
                # Ejecuta la combinación de video y audio en un hilo separado
                threading.Thread(target=combine_video_audio, args=(video_filename, audio_filename, output_filename)).start()
                time.sleep(18)  # Pausa de 30 segundos después de la conversión

        # Muestra el frame con el mensaje de alerta
        cv2.imshow("Camara", frame1)

        # Actualiza los frames
        frame1 = frame2
        ret, frame2 = cap.read()

        # Salir con 'q'
        if cv2.waitKey(10) == ord('1'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()