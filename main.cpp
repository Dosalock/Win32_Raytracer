
#ifndef UNICODE
#define UNICODE
#endif 

#include "main.h"

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
Vect3D D = {};
Vect3D N = {};
Vect3D P = {};
const Vect3D O = { 0,0,0 };
Sphere scene[4] = {};
Light lights[3] = {};

HBITMAP hBitmap = NULL;
HDC hdcWindow = NULL;
BYTE *lpvBits = NULL;
RECT window = {};
int height = 0;
int width = 0;



void CreateScene()
{
	scene[0].center = Vect3D(0, -1, 3);
	scene[0].radius = 1;
	scene[0].color = RGB(255, 0, 0);


	scene[1].center = Vect3D(2, 0, 4);
	scene[1].radius = 1;
	scene[1].color = RGB(0, 0, 255);

	scene[2].center = Vect3D(-2, 0, 4);
	scene[2].radius = 1;
	scene[2].color = RGB(0, 255, 0);

	scene[3].center = Vect3D(0, -5001, 0);
	scene[3].radius = 5000;
	scene[3].color = RGB(255, 255, 0);

	lights[0].type = lights->AMBIENT;
	lights[0].intensity = 0.2;
	//lights[0].pos = { 0,0,0 }; //prettysure this is useless

	lights[1].type = lights->POINT;
	lights[1].intensity = 0.6;
	lights[1].pos = { 2, 1, 0 };

	lights[2].type = lights->DIRECTIONAL;
	lights[2].intensity = 0.6;
	lights[2].pos = { 1, 4, 4 };

	
}
double CalcLight()
{
	double intensity = 0.0;
	Vect3D L = {};
	for (int i = 0; i < sizeof(lights) / sizeof(Light); i++)
	{
		if (lights[i].type == lights->AMBIENT)
		{
			intensity += lights[i].intensity;
		}
		else
		{
			if (lights[i].type == lights->POINT)
			{
				L = (lights[i].pos - P);
			}
			else
			{
				L = lights[i].pos;
			}
			
			double n_dot_l = N.dot(L);
			if (n_dot_l > 0)
			{
				intensity += lights[i].intensity * n_dot_l/(N.len() * L.len());
			}
			
		}
	}
	return intensity;
}
void Draw();
void init()
{
	CreateScene();
    width = window.right;
    height = window.bottom;

    BITMAPINFO bmi = {};
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = width;
    bmi.bmiHeader.biHeight = -height; // Negative to have a top-down DIB
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32;
    bmi.bmiHeader.biCompression = BI_RGB;

    // Create the DIB section and obtain a pointer to the pixel buffer
    hBitmap = CreateDIBSection(NULL, &bmi, DIB_RGB_COLORS, (void**)&lpvBits, NULL, 0);

    if (!lpvBits || !hBitmap) {
        MessageBox(NULL, L"Could not allocate memory for bitmap", L"Error", MB_OK | MB_ICONERROR);
        exit(1);
    }

    // Initialize all pixels to black
    memset(lpvBits, 0, width * height * 4);


}

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

		CW_USEDEFAULT, CW_USEDEFAULT, 800, 800,

		NULL,
		NULL,
		hInstance,
		NULL
		);

	if (hwnd == NULL)
	{
		return 0;
	}

	GetClientRect(hwnd, &window);
	init();
	ShowWindow(hwnd, nCmdShow);
	UpdateWindow(hwnd);

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

	
	width = window.right;
	height = window.bottom;
	switch (uMsg)
	{
		case WM_CREATE:
		{
			


			break;
		}
		case WM_DESTROY:
		{
			// Cleanup
			if (hBitmap) {
                DeleteObject(hBitmap);
            }
			PostQuitMessage(0);
		
			break;
		}

		case WM_PAINT:
		{
			PAINTSTRUCT ps;
			
			GetClientRect(hwnd, &window); // Use client area dimensions
			width = window.right;
			height = window.bottom;
			HDC hdc = BeginPaint(hwnd, &ps);
			
			HDC hdcMem = CreateCompatibleDC(hdc);
			HGDIOBJ oldBitmap = SelectObject(hdcMem, hBitmap);
			
			BitBlt(hdc, 0, 0, width, height, hdcMem, 0, 0, SRCCOPY);

			SelectObject(hdcMem, oldBitmap);
			DeleteDC(hdcMem);

			// All painting occurs here, between BeginPaint and EndPaint.

			EndPaint(hwnd, &ps);
		
			break;	
		}
		case WM_MOVE:
		{
			// TODO: do we want to do something on move?
		}
		case WM_KEYDOWN:
		{
			if (wParam == 'Q') {
				Draw();
				InvalidateRect(hwnd, NULL, TRUE);
			}
			break;
		}
		default:
		{
			return DefWindowProc(hwnd, uMsg, wParam, lParam);
		}
	}
}
QuadraticAnswer IntersectRaySphere(Vect3D O_TEMPCHANGE, Vect3D D, Sphere sphere)
{
	//int r = sphere.radius // is always 1
	double t1, t2;

	Vect3D CO = {};
	CO = O - sphere.center;

	double a = D.dot(D);
	double b = 2 * CO.dot(D);
	double c = CO.dot(CO) - sphere.radius * sphere.radius;  

	double discr = b * b - 4 * a * c;

	if (discr < 0)
	{
		return QuadraticAnswer(INFINITY, INFINITY); 
	}
	
	t1 = (-b + sqrt(discr)) / (2 * a);
	t2 = (-b - sqrt(discr)) / (2 * a);
    return QuadraticAnswer(t1, t2);
}

Vect3D CanvasToViewport(int x, int y, int width, int height) 
{
	// for simplicity : Vw = Vh = d = 1    approx 53 fov
	double aspectRatio = static_cast<double>(width) / height;
    
    // Map x and y to the viewport, adjusting by aspect ratio
    double viewportX = (x - width / 2.0) * (1.0 / width) * aspectRatio;
    double viewportY = -(y - height / 2.0) * (1.0 / height); // Flip Y to match 3D space orientation

    return Vect3D(viewportX, viewportY, 1);  // Z=1 for perspective projection
	//return Vect3D(x * 1.0 / width, y * 1.0 / height, 1);
}
COLORREF TraceRay(Vect3D O_TEMPCHANGE, Vect3D D) 
{
	 double closest_t = INFINITY;
	 Sphere *closest_sphere = NULL;

	 for (int i = 0; i != (sizeof(scene) / sizeof(Sphere)); i++)
	 {
		 QuadraticAnswer res = IntersectRaySphere(O, D, scene[i]);
		 double t1 = res.t1;
		 double t2 = res.t2;
		 
		
		 if (t1 > 1e-5  && t1 < closest_t)
		 {
			 closest_t = t1;
			 closest_sphere = &scene[i];
		 }
		 if (t2 > 1e-5 && t2 < closest_t)
		 {
			 closest_t = t2;
			 closest_sphere = &scene[i];
		 }
	 }
	 if (closest_sphere == NULL)
	 {
		 return RGB(255, 255, 255);
	 }


	 P = O + (D * closest_t);
	 N = P - closest_sphere->center;
	 N = N / N.len();

	 double res = CalcLight();
	 int r = static_cast<int>(GetRValue(closest_sphere->color) + 0.5) * res;
	 int g = static_cast<int>(GetGValue(closest_sphere->color) + 0.5) * res;
	 int b = static_cast<int>(GetBValue(closest_sphere->color) + 0.5) * res;

	 return RGB(r, g, b);
}

void Draw()
{	
	for (int x = 0; (x < (width)); ++x)
	{
		for (int y = 0; (y < (height)); ++y)
		{
			D = CanvasToViewport(x, y, width, height);
			COLORREF color = TraceRay(O, D);
		 
			
			int offset = (y * width + x) * 4;
			if (offset >= 0 && offset < width * height * 4 - 4) {
				lpvBits[offset + 0] = (int)GetBValue(color);
				lpvBits[offset + 1] = (int)GetGValue(color);
				lpvBits[offset + 2] = (int)GetRValue(color);
				lpvBits[offset + 3] = 255;
			}
		
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



