/*------------------Libraries---------------------*/
#include "raytrace.h"

/*------------Varible initialzation---------------*/

Vect3D D = {};
Vect3D N = {};
Vect3D P = {};
const Vect3D O = { 0,0,0 };
Sphere scene[4] = {};
Light lights[3] = {};

/*------------Funcition Defenitions---------------*/

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
	lights[2].intensity = 0.2;
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
				L = (lights[i].pos - P).norm();
			}
			else
			{
				L = lights[i].pos.norm();
			}

			double n_dot_l = N.dot(L);
			if (n_dot_l > 0)
			{
				intensity += lights[i].intensity * n_dot_l / (N.len() * L.len());
			}

		}
	}
	return intensity;
}

void Init(BYTE** pLpvBits, RECT* window, HBITMAP* pHBitmap)
{
	CreateScene();
	int width = (*window).right;
	int height = (*window).bottom;

	BITMAPINFO bmi = {};
	bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
	bmi.bmiHeader.biWidth = width;
	bmi.bmiHeader.biHeight = -height; // Negative to have a top-down DIB
	bmi.bmiHeader.biPlanes = 1;
	bmi.bmiHeader.biBitCount = 32;
	bmi.bmiHeader.biCompression = BI_RGB;

	// Create the DIB section and obtain a pointer to the pixel buffer
	*pHBitmap = CreateDIBSection(NULL, &bmi, DIB_RGB_COLORS, (void**)&(*pLpvBits), NULL, 0);

	if (!(*pLpvBits) || !(*pHBitmap)) {
		MessageBox(NULL, L"Could not allocate memory for bitmap", L"Error", MB_OK | MB_ICONERROR);
		exit(1);
	}

	// Initialize all pixels to black
	memset(*pLpvBits, 0, width * height * 4);


}

QuadraticAnswer IntersectRaySphere(Vect3D D, Sphere sphere)
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
	double fovMod = 1;
	double viewportX = (x - width / 2.0) * ((1.0 * fovMod) / width) * aspectRatio;
	double viewportY = -(y - height / 2.0) * ((1.0 * fovMod) / height); // Flip Y to match 3D space orientation

	return Vect3D(viewportX, viewportY, 1);  // Z=1 for perspective projection
	//return Vect3D(x * 1.0 / width, y * 1.0 / height, 1);
}

Intersection ClosestIntersection(Vect3D D, double t_min, double t_max)
{
	double closest_t = INFINITY;
	Sphere *closest_sphere = NULL;
	
	for (auto &x : scene)
	{
		QuadraticAnswer res = IntersectRaySphere(D, x);
	
		if (res.t1 > 0 && res.t1 < closest_t)
		{
			closest_t = res.t1;
			closest_sphere = const_cast<Sphere*>(&x);
		}
		if (res.t2 > 0 && res.t2 < closest_t)
		{
			closest_t = res.t2;
			closest_sphere = const_cast<Sphere *>(&x);;
		}
	}
	return Intersection(closest_sphere, closest_t);
}

COLORREF TraceRay(Vect3D D)
{
	N = {};
	P = {};

	//double closest_t = INFINITY;
	//Sphere* closest_sphere = NULL;

	auto [closest_sphere, closest_t] = ClosestIntersection(D, 0, 0);
	if (closest_sphere == NULL)
	{
		return RGB(255, 255, 255);
	}


	P = O + (D * closest_t);
	N = (P - closest_sphere->center).norm();
	N = N / N.len(); /* godly error correcion */


	double res = CalcLight();
	int r = (int)round(GetRValue(closest_sphere->color) * res);
	int g = (int)round(GetGValue(closest_sphere->color) * res);
	int b = (int)round(GetBValue(closest_sphere->color) * res);
	

	return RGB(r, g, b);
}

void Draw(BYTE** pLpvBits, int width, int height)
{
	
	for (int x = 0; (x < (width)); ++x)
	{
		for (int y = 0; (y < (height)); ++y)
		{
			D = CanvasToViewport(x, y, width, height);
			COLORREF color = TraceRay(D);


			int offset = (y * width + x) * 4;
			if (offset >= 0 && offset < width * height * 4 - 4) {
				(*pLpvBits)[offset + 0] = (int)GetBValue(color);
				(*pLpvBits)[offset + 1] = (int)GetGValue(color);
				(*pLpvBits)[offset + 2] = (int)GetRValue(color);
				(*pLpvBits)[offset + 3] = 255;
			}

		}
	}
}

