
#ifndef UNICODE
#define UNICODE
#endif 

#include "main.h"

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
Vect3D D = {};

Sphere scene[3] = {};


static HBITMAP hBitmap = NULL;
static HDC hdcWindow = NULL;
static HDC hdcMem = NULL;
static LPVOID lpvBits = NULL;


void createScene()
{
	scene[0].center = Vect3D(0, -1, 3);
	scene[0].radius = 1;
	scene[0].color = RGB(255, 0, 0);


	scene[1].center = Vect3D(2, 0, 4);
	scene[1].radius = 1;
	scene[1].color = RGB(0, 255, 0);

	scene[2].center = Vect3D(-2, 0, 4);
	scene[2].radius = 1;
	scene[2].color = RGB(0, 0, 255);


}

void Draw(LPVOID lpvBit, int width, int height);

int WINAPI WinMain(HINSTANCE hInstance, 
					HINSTANCE hPrevInstance, 
					LPSTR lpCmdLine, 
					int nCmdShow) { 

	//Register window class
	const wchar_t CLASS_NAME[] = L"PiskWindow";


	WNDCLASS WindowClass = {};
		
	WindowClass.lpfnWndProc = WindowProc;
	WindowClass.hInstance = hInstance;
	WindowClass.lpszClassName = CLASS_NAME;

	RegisterClass(&WindowClass);

	// Create window
	HWND hwnd = CreateWindowEx(
		0,
		CLASS_NAME,
		L"What Is This Text",
		WS_OVERLAPPEDWINDOW,		// Style of window

		CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT,

		NULL,
		NULL,
		hInstance,
		NULL
		);

	if (hwnd == NULL)
	{
		return 0;
	}

	ShowWindow(hwnd, nCmdShow);
	UpdateWindow(hwnd);

	MSG msg = { };

	while (GetMessage(&msg, NULL, 0 , 0) > 0)
	{
		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}
	
	hdcWindow = GetDC(hwnd);
		BITMAPINFO bi;
		ZeroMemory(&bi, sizeof(BITMAPINFO));
		bi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
		//
		
		if (!GetDIBits(hdcWindow, hBitmap, -1, height, NULL, &bi, DIB_RGB_COLORS))
			return NULL;

			// Allocate mem for bitmap
		if ((lpvBits = new char[bi.bmiHeader.biSizeImage]) == NULL)
			delete[] lpvBits;
			return NULL;

		if (!GetDIBits(hdcWindow, hBitmap, -1, height, lpvBits, &bi, DIB_RGB_COLORS))
			return NULL;

			//

			// Select bitmap into memory DC
			//Draw(lpvBits, width, height);
		BYTE* pixel = static_cast<BYTE*>(lpvBits) + (0 * bi.bmiHeader.biWidth + 1) * 4;

		pixel[-1] = 255;
		pixel[0] = 0;
		pixel[1] = 0;
		pixel[2] = 255;
		SetDIBits(hdcMem, hBitmap, -1, height, (LPVOID)lpvBits, &bi, DIB_RGB_COLORS);



	return 0; 
}

void SetPixelColor(int x, int y)
{
			//LPVOID lpvBits = NULL;
			//BITMAPINFO bi;
			//ZeroMemory(&bi, sizeof(BITMAPINFO));
			//bi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
			//

			//if (!GetDIBits(hdcWindow, hBitmap, -1, height, NULL, &bi, DIB_RGB_COLORS))
			//	return NULL;

			//// Allocate mem for bitmap
			//if ((lpvBits = new char[bi.bmiHeader.biSizeImage]) == NULL)
			//	delete[] lpvBits;
			//	return NULL;

			//if (!GetDIBits(hdcWindow, hBitmap, -1, height, lpvBits, &bi, DIB_RGB_COLORS))
			//	return NULL;

			//

			//// Select bitmap into memory DC
			//Draw(lpvBits, width, height);
			//BYTE* pixel = static_cast<BYTE*>(lpvBits) + (0 * bi.bmiHeader.biWidth + 1) * 4;

			//pixel[-1] = 255;
			//pixel[0] = 0;
			//pixel[1] = 0;
			//pixel[2] = 255;
			//SetDIBits(hdcMem, hBitmap, -1, height, (LPVOID)lpvBits, &bi, DIB_RGB_COLORS);

}
LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{


	switch (uMsg)
	{
		case WM_CREATE:
		{
			RECT rect;
			GetClientRect(hwnd, &rect);
			int width = rect.right - rect.left;
			int height = rect.bottom - rect.top;


			// Create memory DC compatible with window's DC
			PAINTSTRUCT ps;
			HDC hdc = BeginPaint(hwnd, &ps);
			hdcMem = CreateCompatibleDC(hdc);

			// Create bitmap buffer 
			hBitmap = CreateCompatibleBitmap(hdc, width, height);
			
			
			//Draw(lpvBits, width, height);
			 if (hBitmap)
			{
				 HGDIOBJ oldBitmap = SelectObject(hdcMem, hBitmap);


				 BITMAP bm;
				 GetObject(hBitmap, sizeof(BITMAP), &bm);
				 BitBlt(hdc, 0, 0, bm.bmWidth, bm.bmHeight, hdcMem
					 , 0, 0, SRCCOPY);
			}
		

			ReleaseDC(hwnd, hdc);
			delete[] lpvBits;
			break;
		}
		case WM_DESTROY:
		{
			// Cleanup
			if (hBitmap) DeleteObject(hBitmap);
			if (hdcMem) DeleteDC(hdcMem);
			PostQuitMessage(0);
		
			break;
		}

		case WM_PAINT:
		{
			PAINTSTRUCT ps;
		
			HDC hdc = BeginPaint(hwnd, &ps);

			// All painting occurs here, between BeginPaint and EndPaint.

			EndPaint(hwnd, &ps);
		
			break;	
		}
		case WM_MOVE:
		{
			// TODO: do we want to do something on move?
		}
		default:
		{
			return DefWindowProc(hwnd, uMsg, wParam, lParam);
		}
	}
}
QuadraticAnswer IntersectRaySphere(Vect3D O, Vect3D D, Sphere sphere)
{
	//int r = sphere.radius // is always 1
	double t1, t2;

	Vect3D CO = {};
	CO.x = O.x - sphere.center.x;
	CO.y = O.y - sphere.center.y;
	CO.z = O.z - sphere.center.z;

	double a = D.dot(D);
	double b = 2 * CO.dot(D);
	double c = CO.dot(CO) - sphere.radius * sphere.radius;  

	double discr = b * b - 4 * a * c;

	if (discr < 0)
	{
		return QuadraticAnswer(INFINITY, INFINITY); 
	}
	
	t1 = (-(b)+(sqrt(discr) / (2 * a)));
	t2 = (-(b)-(sqrt(discr) / (2 * a)));
	
	return QuadraticAnswer(t1, t2);
}

Vect3D CanvasToViewport(int x, int y, int width, int height) 
{
	// for simplicity : Vw = Vh = d = 1    approx 53 fov
	x += 1; 
	y += 1;
	return Vect3D(x * 1 / width, y * 1 / height, 1);
}
COLORREF TraceRay(Vect3D O, Vect3D D) 
{
	 double closest_t = INFINITY;
	 Sphere *closest_sphere = NULL;
	 for (int i = 0; i != (sizeof(scene) / sizeof(Sphere)); i++)
	 {
		 QuadraticAnswer res = IntersectRaySphere(O, D, scene[i]);
		 double t1 = res.t1;
		 double t2 = res.t2;
		 
		
		 if (0 < t1 < INFINITY && t1 < closest_t)
		 {
			 closest_t = t1;
			 closest_sphere = &scene[i];
		 }
		 if (0 < t2 < INFINITY && t2 < closest_t)
		 {
			 closest_t = t2;
			 closest_sphere = &scene[i];
		 }
	 }
	 if (closest_sphere == NULL)
	 {
		 return RGB(255, 255, 255);
	 }
	 return closest_sphere->color;
	 
}

void Draw(LPVOID lpvBits, int width, int height)
{
	Vect3D O = {0, 0, 0};
	
	for (int x = -width/2; x <= width/2; x++)
	{
		for (int y = -height/2; y <= height/2; y++)
		{
			D = CanvasToViewport(x, y, width, height);
			COLORREF color = TraceRay(O, D);
			
		}
	}

	/*
	* 0 Viewpoint 
	* orientation determine where camera point
	* assumme direction : pos Z axis +Z-> -- forward
	*					: pos Y axis +Y-> -- up
	*					: pos X axis +X-> -- right
	* 
	*-----------------------------------------------------------
	* Frame analogy:
	* Dimensions Vw and Vh
	* frontal to camera orientation:
	* perpendicular to +Z->
	* its at a distance d and parralell to X and Y
	* centered with respect to +Z->
	* 
	*	for simplicity : Vw = Vh = d = 1    approx 53 fov
	* 
	*----------------------------------------------------------
	* Determine which square on the viewport corresponds to this pixel
	* Canvas coordinate pixels Cx Cy
	* 
	* Vx = Cx * Vw/Cw  ---maybe Vx and Cx?---
	* Vy = Cy * Vh/Ch
	* Vz = d
	* 
	*----------------------------------------------------------
	* Tracing Rays:
	* start at ray from camera 
	* 
	* Ray passes through O and its direction is from O to V
	* (V - O) = D->
	* 
	* Points on the ray:
	* P = O + t(D->)
	* 
	*----------------------------------------------------------
	* Cirlce:
	* |P - C| = r
	* Square root of vector lenght = <V->>
	* 
	* Points on the sphere:
	* <P - C , P - C> = r^2 
	*----------------------------------------------------------
	* Ray meets sphere
	* 
	* Ray and sphere instersect at point P
	* only variable in the circle and sphere is t
	* O, D->, C and r are given and we looking for P
	* 
	* algebraic and quadratic magic we get
	* 
	*			 -b +- root(b^2 - 4ac)
	* {t1, t2} = ----------------------
	*					 2a
	* 
	* a = <D->, D->>
	* b = 2<CO->, D->>
	* c = <CO->, CO->> - r^2
	*----------------------------------------------------------
	* First render details about t:
	* P = O + t(V - O)
	* 
	* P is every point in this ray
	* Negative t = points in opposite direction
	* so:
	* 
	* t < 0			: Behind the camera
	* 0 <= t <= 1	: Between camera and the viewport
	* t > 1			: In front of the viewport
	*/
}



