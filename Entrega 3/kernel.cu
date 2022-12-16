/**
* ARQUITECTURA DE COMPUTADORES
* 2º Grado en Ingenieria Informatica
*
* Entrega 3
*
* Alumno: Rodrigo Pascual Arnaiz y Villar Solla, Alejandro
* Fecha: 30/11/2022
*
*/

///////////////////////////////////////////////////////////////////////////
// includes
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "./gpu_bitmap.h"

///////////////////////////////////////////////////////////////////////////
// prototipos

__host__ void leerBMP_RGBA(const char* nombre, int* w, int* h, unsigned char** imagen);
__host__ int mcd(int x, int y);
__global__ void convertirRgbBancoyNegro(unsigned char* colores);

///////////////////////////////////////////////////////////////////////////

// MAIN: rutina principal ejecutada en el host
int main(int argc, char** argv)
{

	// Declaramos las variables
	unsigned char* host_bitmap, * dev_bitmap;
	int ancho, alto;
	int thread_x_tam, thread_y_tam, divisor;
	float elapsedTime;


	// Leemos la imagen con la funcion proporcionada en los apuntes 
	leerBMP_RGBA("imagen.bmp", &ancho, &alto, &host_bitmap);


	// Obtener utilizando el MCD el tamañlo que van a tener los hilos 
	divisor = mcd(ancho, alto);
	thread_x_tam = ancho / divisor;
	thread_y_tam = alto / divisor;

	// Declaracion del bitmap:
	// Inicializacion de la estructura RenderGPU
	RenderGPU foto(ancho, alto);

	// Tamaño del bitmap en bytes
	size_t size = foto.image_size();

	// Asignacion y reserva de la memoria en el host (framebuffer) 
	unsigned char* host_imagen = foto.get_ptr();

	// Reservamos el hueco del dev bitmap
	cudaMalloc((void**)&dev_bitmap, size);


	// Movemos el bitmap del host a device 
	cudaMemcpy(dev_bitmap, host_bitmap, size, cudaMemcpyHostToDevice);

	// Se calula el numero de bloques que se va a necesitar por cada hilo, para ello 
	// dividimos el ancho o el alto entre el tamaño de los hilos que le corresponda 
	dim3 Nbloques(ancho / thread_x_tam, alto / thread_y_tam);

	// Definimos los hilos teniendo ecuenta los tamaños calculados antes para cada hilo
	// usamos la variable thread_x_tam, thread_y_tam
	dim3 hilosB(thread_x_tam, thread_y_tam);

	// Inicializamos los eventos 
	cudaEvent_t inicio, fin;
	cudaEventCreate(&inicio);
	cudaEventCreate(&fin);

	// El evento de inicio lo ponemos a 0
	cudaEventRecord(inicio, 0);

	// Lanzamos la función kernel que va a convertir todos los pixeles a banco y negro
	convertirRgbBancoyNegro << <Nbloques, hilosB >> > (dev_bitmap);;

	// Registramos el evento de FIN como 0 y sincronizamos los eventos 
	cudaEventRecord(fin, 0);
	cudaEventSynchronize(fin);

	// Recogemos el bitmap desde la GPU para visualizarlo
	cudaMemcpy(host_imagen, dev_bitmap, size, cudaMemcpyDeviceToHost);

	// Calculamos la elipsis de tiempo transcurrido
	cudaEventElapsedTime(&elapsedTime, inicio, fin);

	// visualizamos el tamaño del kernel 
	printf("\nTamño del kernel: " );
	printf("\nTamño numero de bloques en x:	%i -> con %i hilos", Nbloques.x, ( hilosB.x * hilosB.y) );
	printf("\nTamño numero de bloques en y:	%i -> con %i hilos", Nbloques.y, (hilosB.x * hilosB.y));
	printf("\nTamño numero de hilos en x:	%i", hilosB.x);
	printf("\nTamño numero de hilos en y:	%i", hilosB.y);
	printf("\nTotal de: %i",  (hilosB.x * hilosB.y) * Nbloques.x  * Nbloques.y );


	// Visualizacion y salida
	printf("\nEl tiempo de procesamiento ha sido de %f ms", elapsedTime);

	printf("\n...pulsa [ESC] para finalizar...");


	// Destruimos todos los eventos 
	cudaEventDestroy(inicio);
	cudaEventDestroy(fin);

	// Visualización de la foto pasada a negro 
	foto.display_and_exit();


	// Fin del programa
	return 0;
}

/**
* Funcion: leerBMP_RGBA ( HOST )
* Objetivo: Función que se encarga de leer un archivo de BMP
*
* @param const char* nombre -> Nombre del BMP
* @param int* w -> Ancho de la imagen en pixeles
* @param int* h -> Alto de la imagen en pixeles
* @param unsigned char** imagen -> Puntero al array de datos de la imagen en formato RGBA
* @return: void
*/
__host__ void leerBMP_RGBA(const char* nombre, int* w, int* h, unsigned char** imagen)
{
	// Lectura del archivo .BMP
	FILE* archivo;

	// Abrimos el archivo en modo solo lectura binaria
	if ((archivo = fopen(nombre, "rb")) == NULL)
	{
		printf("\nERROR ABRIENDO EL ARCHIVO %s...", nombre);
		// salida
		printf("\npulsa [INTRO] para finalizar");
		getchar();
		exit(1);
	}
	printf("> Archivo [%s] abierto:\n", nombre);

	// En Windows, la cabecera tiene un tamaño de 54 bytes:
	// 14 bytes (BMP header) + 40 bytes (DIB header)
	// BMP HEADER
	// Extraemos cada campo y lo almacenamos en una variable del tipo adecuado
	// posición 0x00 -> Tipo de archivo: "BM" (leemos 2 bytes)
	unsigned char tipo[2];
	fread(tipo, 1, 2, archivo);

	// Comprobamos que es un archivo BMP
	if (tipo[0] != 'B' || tipo[1] != 'M')
	{
		printf("\nERROR: EL ARCHIVO %s NO ES DE TIPO BMP...", nombre);
		// salida
		printf("\npulsa [INTRO] para finalizar");
		getchar();
		exit(1);
	}

	// posición 0x02 -> Tamaño del archivo .bmp (leemos 4 bytes)
	unsigned int file_size;
	fread(&file_size, 4, 1, archivo);

	// posición 0x06 -> Campo reservado (leemos 2 bytes)
	// posición 0x08 -> Campo reservado (leemos 2 bytes)
	unsigned char buffer[4];
	fread(buffer, 1, 4, archivo);

	// posición 0x0A -> Offset a los datos de imagen (leemos 4 bytes)
	unsigned int offset;
	fread(&offset, 4, 1, archivo);

	// imprimimos los datos
	printf(" \nDatos de la cabecera BMP\n");
	printf("> Tipo de archivo : %c%c\n", tipo[0], tipo[1]);
	printf("> Tamano del archivo : %u KiB\n", file_size / 1024);
	printf("> Offset de datos : %u bytes\n", offset);

	// DIB HEADER
	// Extraemos cada campo y lo almacenamos en una variable del tipo adecuado
	// posición 0x0E -> Tamaño de la cabecera DIB (BITMAPINFOHEADER) (leemos 4bytes)
	unsigned int header_size;
	fread(&header_size, 4, 1, archivo);

	// posición 0x12 -> Ancho de la imagen (leemos 4 bytes)
	unsigned int ancho;
	fread(&ancho, 4, 1, archivo);

	// posición 0x16 -> Alto de la imagen (leemos 4 bytes)
	unsigned int alto;
	fread(&alto, 4, 1, archivo);

	// posición 0x1A -> Numero de planos de color (leemos 2 bytes)
	unsigned short int planos;
	fread(&planos, 2, 1, archivo);

	// posición 0x1C -> Profundidad de color (leemos 2 bytes)
	unsigned short int color_depth;
	fread(&color_depth, 2, 1, archivo);

	// posicion 0x1E -> Tipo de compresión (leemos 4 bytes)
	unsigned int compresion;
	fread(&compresion, 4, 1, archivo);

	// imprimimos los datos
	printf(" \nDatos de la cabecera DIB\n");
	printf("> Tamano de la cabecera: %u bytes\n", header_size);
	printf("> Ancho de la imagen : %u pixeles\n", ancho);
	printf("> Alto de la imagen : %u pixeles\n", alto);
	printf("> Planos de color : %u\n", planos);
	printf("> Profundidad de color : %u bits/pixel\n", color_depth);
	printf("> Tipo de compresion : %s\n", (compresion == 0) ? "none" : "unknown");

	// LEEMOS LOS DATOS DEL ARCHIVO
	// Calculamos espacio para una imagen de tipo RGBA:
	size_t img_size = ancho * alto * 4;

	// Reserva para almacenar los datos del bitmap
	unsigned char* datos = (unsigned char*)malloc(img_size);

	// Desplazamos el puntero FILE hasta el comienzo de los datos de imagen: 0 +offset
	fseek(archivo, offset, SEEK_SET);

	// Leemos píxel a pixel, reordenamos (BGR -> RGB) e insertamos canal alfa
	unsigned int pixel_size = color_depth / 8;
	for (unsigned int i = 0; i < ancho * alto; i++)
	{
		fread(buffer, 1, pixel_size, archivo); // leemos el pixel i
		datos[i * 4 + 0] = buffer[2]; // escribimos canal R
		datos[i * 4 + 1] = buffer[1]; // escribimos canal G
		datos[i * 4 + 2] = buffer[0]; // escribimos canal B
		datos[i * 4 + 3] = buffer[3]; // escribimos canal alfa (si lo hay)
	}
	// Cerramos el archivo
	fclose(archivo);

	// PARAMETROS DE SALIDA
	// Ancho de la imagen en pixeles
	*w = ancho;

	// Alto de la imagen en pixeles
	*h = alto;

	// Puntero al array de datos RGBA
	*imagen = datos;

	// Salida
	return;
}

/**
* Funcion: convertirRgbBancoyNegro ( GLOBAL )
* Objetivo: Función que convierte una matriz de colores RGB a
*	blanco y negro, formula:
*	Y = 0.299×R + 0.587×G + 0.114×B
*
* @param unsigned char *colores -> array de colores
* @return: void
*/
__global__ void convertirRgbBancoyNegro(unsigned char* imagen)
{
	// coordenada vertical de cada hilo
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	// coordenada horizontal de cada hilo
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	// coordenada global de cada hilo (indice para acceder a la memoria)
	int pos = x + y * blockDim.x * gridDim.x;

	// cada hilo obtiene la posicion de un pixel
	int pixel = pos * 4;

	// Calculamos el tono de gris con la fórmula
	int Y = 0.299F * imagen[pixel + 0] + 0.587F * imagen[pixel + 1] + 0.114F * imagen[pixel + 2];

	// Establecemos los nuevos valores de color en el RGB
	imagen[pixel + 0] = Y;
	imagen[pixel + 1] = Y;
	imagen[pixel + 2] = Y;

}

/**
* Función que retorna el maximo comun divisor 
*
* @param int x -> primner valor
* @param int 2 -> segundo valor 
* @return int -> resultado 
*/
__host__ int mcd(int x, int y)
{
	return y ? mcd(y, x % y) : x;
	 
}