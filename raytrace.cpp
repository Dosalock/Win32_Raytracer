/**
 *
 *  @file      raytrace.cpp
 *  @brief     Declaration of core raytracing functionality
 *  @author    Johan Karlsson - github.com/dosalock
 *  @date      8.11.2024
 *  @copyright Copyright © [2024] [Johan Karlsson]
 *
 */


/*------------------Libraries---------------------*/
#include "raytrace.h"

/*------------Varible initialzation---------------*/



/*------------Funcition Defenitions---------------*/

/**
 * @brief Checks if a root is within limits and is a better match than
 * previously cached
 * @param root - Root to check
 * @param t_min - Minimum acceptable value for root
 * @param t_max - Maximum acceptable value for root
 * @param closest_t - Current best root
 * @return
 */
bool IsBetterRoot ( float root, float t_min, float t_max, float closest_t )
{
	return IsInBounds( root, t_min, t_max ) && root < closest_t;
}

Vect3D ReflectRay ( Vect3D R, Vect3D N )
{
	return ( ( N * ( N.dot( R ) ) ) * 2 ) - R;
}

void CreateScene ( Sphere *scene, Light *lights )
{
	scene[0].center      = Vect3D( 0, -1, 3 );
	scene[0].radius      = 1;
	scene[0].color       = RGB( 255, 0, 0 );
	scene[0].specularity = 500;
	scene[0].reflective  = 0.2;
	scene[0].raidus_squared     = scene[0].radius * scene[0].radius;

	scene[1].center      = Vect3D( 2, 0, 4 );
	scene[1].radius      = 1;
	scene[1].color       = RGB( 0, 0, 255 );
	scene[1].specularity = 500;
	scene[1].reflective  = 0.3;
	scene[1].raidus_squared     = scene[1].radius * scene[1].radius;

	scene[2].center      = Vect3D( -2, 0, 4 );
	scene[2].radius      = 1;
	scene[2].color       = RGB( 0, 255, 0 );
	scene[2].specularity = 10;
	scene[2].reflective  = 0.4;
	scene[2].raidus_squared     = scene[2].radius * scene[2].radius;

	scene[3].center      = Vect3D( 0, -5001, 0 );
	scene[3].radius      = 5000;
	scene[3].color       = RGB( 255, 255, 255 );
	scene[3].specularity = 1000;
	scene[3].reflective  = 0.5;
	scene[3].raidus_squared     = scene[3].radius * scene[3].radius;

	lights[0].type      = lights->AMBIENT;
	lights[0].intensity = 0.2;
	// lights[0].pos = { 0,0,0 }; //prettysure this is useless

	lights[1].type      = lights->POINT;
	lights[1].intensity = 0.6;
	lights[1].pos       = { 2, 1, 0 };

	lights[2].type      = lights->DIRECTIONAL;
	lights[2].intensity = 0.2;
	lights[2].pos       = { 1, 4, 4 };
}

float CalcLight ( Vect3D P, Vect3D N, Vect3D V, int s, Sphere* scene, Light *lights)
{
	float intensity = 0.0;
	float t_max     = 0;
	Vect3D L         = { };
	Vect3D R         = { };
	for ( int i = 0; i < 4; i++ )
	{
		if ( lights[i].type == lights->AMBIENT )
		{
			intensity += lights[i].intensity;
		}
		else
		{
			if ( lights[i].type == lights->POINT )
			{
				L     = ( lights[i].pos - P );
				t_max = 1;
			}
			else
			{
				L     = lights[i].pos;
				t_max = INFINITY;
			}
			float t_min = 0.0001f;
			auto [shadow_sphere, shadow_t] =
				ClosestIntersection( P, L, t_min, t_max, scene);
			if ( shadow_sphere != NULL )
			{
				continue;
			}

			float n_dot_l = N.dot( L );
			if ( n_dot_l > 0 )
			{
				intensity +=
					lights[i].intensity * n_dot_l / ( N.len( ) * L.len( ) );
			}

			if ( s != -1 )
			{
				R = ReflectRay( L, N );

				float r_dot_v = R.dot( V );


				if ( r_dot_v > 0 )
				{
					intensity +=
						lights[i].intensity
						* pow( r_dot_v / ( R.len( ) * ( V.len( ) ) ), s );
				}
			}
		}
	}
	return intensity;
}

void Init ( BYTE **pLpvBits, RECT *window, HBITMAP *pHBitmap )
{
	
	int width  = ( *window ).right;
	int height = ( *window ).bottom;

	BITMAPINFO bmi              = { };
	bmi.bmiHeader.biSize        = sizeof( BITMAPINFOHEADER );
	bmi.bmiHeader.biWidth       = width;
	bmi.bmiHeader.biHeight      = -height; // Negative to have a top-down DIB
	bmi.bmiHeader.biPlanes      = 1;
	bmi.bmiHeader.biBitCount    = 32;
	bmi.bmiHeader.biCompression = BI_RGB;

	// Create the DIB section and obtain a pointer to the pixel buffer
	*pHBitmap = CreateDIBSection( NULL,
								  &bmi,
								  DIB_RGB_COLORS,
								  ( void ** )&( *pLpvBits ),
								  NULL,
								  0 );

	if ( !( *pLpvBits ) || !( *pHBitmap ) )
	{
		MessageBox( NULL,
					L"Could not allocate memory for bitmap",
					L"Error",
					MB_OK | MB_ICONERROR );
		exit( 1 );
	}

	// Initialize all pixels to black
	memset( *pLpvBits, 0, width * height * 4 );
}

float
	IntersectRaySphere ( Vect3D O, Vect3D D, Sphere sphere, float dDot )
{
	Vect3D CO = O - sphere.center;
	float a   = D.dot( D );
	float b   = 2 * CO.dot( D );
	float c   = CO.dot( CO ) - sphere.raidus_squared;

	float discr = b * b - 4 * a * c;

	if ( discr < 0 )
	{
		return INFINITY;
	}
	else if ( discr == 0 )
	{
		return -b / ( 2 * a );
	}

	float t = ( -b - sqrtf( discr ) )
			  / ( 2 * a ); // Minimize compute only go for 1 root;

	return t;
}

Vect3D CanvasToViewport ( int x, int y, int width, int height )
{
	// for simplicity : Vw = Vh = d = 1    approx 53 fov
	float aspect_ratio = static_cast<float>( width ) / height;

	// Map x and y to the viewport, adjusting by aspect ratio
	float fov_mod = 1;
	float viewport_x =
		( x - width / 2.0 ) * ( ( 1.0 * fov_mod ) / width ) * aspect_ratio;
	float viewport_y = -( y - height / 2.0 )
						* ( ( 1.0 * fov_mod )
							/ height ); // Flip Y to match 3D space orientation

	return Vect3D( viewport_x,
				   viewport_y,
				   1 ); // Z=1 for perspective projection
}

Intersection
	ClosestIntersection ( Vect3D O, Vect3D D, float t_min, float t_max, Sphere *scene )
{
	float closest_t        = INFINITY;
	Sphere *closest_sphere = NULL;
	float d_dot_d          = D.dot( D ); // Cache immutable value


	for ( int sphere = 0; sphere < 4; sphere++ )
	{
		float t = IntersectRaySphere( O, D, scene[sphere], d_dot_d );

		if ( IsBetterRoot( t, t_min, t_max, closest_t ) )
		{
			closest_t      = t;
			closest_sphere = const_cast<Sphere *>( &scene[sphere] );
		}
	}
	return Intersection( closest_sphere, closest_t );
}

COLORREF TraceRay ( Vect3D O,
					Vect3D D,
					float t_min,
					float t_max,
					int recursionDepth,
					Sphere *scene,
					Light *lights )
{
	Vect3D N = { };
	Vect3D P = { };
	Vect3D R = { };

	Intersection intersec =
		ClosestIntersection( O, D, t_min, t_max, scene );
	Sphere *closest_sphere = intersec.sphere;
	float closest_t        = intersec.point;
	if ( closest_sphere == NULL )
	{
		return RGB( 0, 0, 0 );
	}

	P = O + ( D * closest_t );
	N = ( P - closest_sphere->center );
	N.norm( );

	float res = CalcLight( P,
						   N,
						   D.invert( ),
						   closest_sphere->specularity,
						   scene,
						   lights );
	int r     = ( int )round( GetRValue( closest_sphere->color ) * res );
	int g     = ( int )round( GetGValue( closest_sphere->color ) * res );
	int b     = ( int )round( GetBValue( closest_sphere->color ) * res );

	float refl = closest_sphere->reflective;

	if ( recursionDepth <= 0 || refl <= 0 )
	{
		return RGB( max( 0, min( 255, r ) ),
					max( 0, min( 255, g ) ),
					max( 0, min( 255, b ) ) );
	}

	R = ReflectRay( D.invert( ), N );
	COLORREF reflectedColor = TraceRay( P,
										R,
										t_min,
										t_max,
										recursionDepth - 1,
										scene,
										lights );

	int reflected_r = ( int )roundf( GetRValue( reflectedColor ) ) * refl;
	int reflected_g = ( int )roundf( GetGValue( reflectedColor ) ) * refl;
	int reflected_b = ( int )roundf( GetBValue( reflectedColor ) ) * refl;


	return RGB(
		max( 0,
			 min( 255, static_cast<int>( r * ( 1 - refl ) + reflected_r ) ) ),
		max( 0,
			 min( 255, static_cast<int>( g * ( 1 - refl ) + reflected_g ) ) ),
		max( 0,
			 min( 255, static_cast<int>( b * ( 1 - refl ) + reflected_b ) ) ) );
}

void Draw ( BYTE **pLpvBits, int width, int height, Camera cam, Sphere *scene, Light *lights)
{
	Vect3D D           = { };
	Vect3D N           = { };
	Vect3D P           = { };
	Vect3D O     = { 0, 0, 0 };
	float t_min       = 0.001;
	float t_max       = INFINITY;
	int recursionDepth = 1;

	for ( int x = 0; ( x < ( width ) ); ++x )
	{
		for ( int y = 0; ( y < ( height ) ); ++y )
		{
			D = CanvasToViewport( x, y, width, height );
			D = cam.ApplyCameraRotation( D, cam );

			COLORREF color =
				TraceRay( cam.position, D, t_min, t_max, recursionDepth, scene, lights );


			D = D.norm();
			int offset = ( y * width + x ) * 4;
			if ( offset >= 0 && offset < width * height * 4 - 4 )
			{
				( *pLpvBits )[offset + 0] = ( int )GetBValue( color );
				( *pLpvBits )[offset + 1] = ( int )GetGValue( color );
				( *pLpvBits )[offset + 2] = ( int )GetRValue( color );
				( *pLpvBits )[offset + 3] = 255;
			}
		}
	}
}
