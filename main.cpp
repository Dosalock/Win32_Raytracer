
#ifndef UNICODE
#define UNICODE
#endif 

#include "main.h"

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
Vect3D D = {};

Sphere scene[3];

void Draw(HDC hdcMem, int width, int height);

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

	MSG msg = { };

	while (GetMessage(&msg, NULL, 0 , 0) > 0)
	{
		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}

	return 0; 
}


LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	static HBITMAP hBitmap = NULL;
	static HDC hdcMem = NULL;

	switch (uMsg)
	{
		case WM_CREATE:
		{
			RECT rect;
			GetClientRect(hwnd, &rect);
			int width = rect.right - rect.left;
			int height = rect.bottom - rect.top;


			// Create memory DC compatible with window's DC
			HDC hdcWindow = GetDC(hwnd);
			hdcMem = CreateCompatibleDC(hdcWindow);

			// Create bitmap buffer 
			hBitmap = CreateCompatibleBitmap(hdcWindow, width, height);

			// Select bitmap into memory DC
			SelectObject(hdcMem, hBitmap);

			ReleaseDC(hwnd, hdcWindow);
			Draw(NULL, NULL, NULL);
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

			BitBlt(hdc,
				0,
				0,
				ps.rcPaint.right - ps.rcPaint.left,
				ps.rcPaint.bottom - ps.rcPaint.top,
				hdcMem,
				0,
				0,
				SRCCOPY);

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

	Vect3D CO = {};
	CO.x = O.x - sphere.center.x;
	CO.y = O.y - sphere.center.y;
	CO.z = O.z - sphere.center.z;

	Vect3D a = { D.x * D.x , D.y * D.y, D.z * D.z};
	Vect3D b = { 2 * CO.x * D.x , 2 * CO.y * D.y, 2 * CO.z * D.z};
	Vect3D c = { CO.x * CO.x , CO.y * CO.y, CO.z * CO.z }/* - r * r  */;
	
	double discr = b.x;
}

Vect3D CanvasToViewport(int x, int y, int width, int height) 
{
	// for simplicity : Vw = Vh = d = 1    approx 53 fov
	return Vect3D(x * 1 / width, y * 1 / height, 1);
}
COLORREF TraceRay(Vect3D O, Vect3D D) 
{
	 double closest_t = INFINITY;
	 pSphere closest_sphere = NULL;
	 for (int i = 0; i != (sizeof(scene) /sizeof(Sphere)); i++)
	 {
		 // t1, t2 = instersectrayspherejadajada
		 double t1, t2;
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

void Draw(HDC hdcMem, int width, int height)
{
	Vect3D O_ViewPoint = {0, 0, 0};
	
	for (int x = -width/2; x <= width/2; x++)
	{
		for (int y = -height/2; y <= height/2; y++)
		{
			D = CanvasToViewport(x, y, width, height);
			/*
			* color = TraceRay (O, D, 1, inf)
			* canvas.PutPixel(x,y, color)
			*/
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



