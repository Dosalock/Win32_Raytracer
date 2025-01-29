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
	scene[0].center         = Vect3D( 0, -1, 3 );
	scene[0].radius         = 1;
	scene[0].color          = RGB( 255, 0, 0 );
	scene[0].specularity    = 500;
	scene[0].reflective     = 0.2;
	scene[0].raidus_squared = scene[0].radius * scene[0].radius;

	scene[1].center         = Vect3D( 2, 0, 4 );
	scene[1].radius         = 1;
	scene[1].color          = RGB( 0, 0, 255 );
	scene[1].specularity    = 500;
	scene[1].reflective     = 0.3;
	scene[1].raidus_squared = scene[1].radius * scene[1].radius;

	scene[2].center         = Vect3D( -2, 0, 4 );
	scene[2].radius         = 1;
	scene[2].color          = RGB( 0, 255, 0 );
	scene[2].specularity    = 10;
	scene[2].reflective     = 0.4;
	scene[2].raidus_squared = scene[2].radius * scene[2].radius;

	scene[3].center         = Vect3D( 0, -5001, 0 );
	scene[3].radius         = 5000;
	scene[3].color          = RGB( 255, 255, 0 );
	scene[3].specularity    = 1000;
	scene[3].reflective     = 0.5;
	scene[3].raidus_squared = scene[3].radius * scene[3].radius;

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

float CalcLight ( Vect3D P,
				  Vect3D N,
				  Vect3D V,
				  int s,
				  Sphere *scene,
				  Light *lights )
{
	float intensity = 0.0;
	float t_max     = 0;
	Vect3D L        = { };
	Vect3D R        = { };
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
				ClosestIntersection( P, L, t_min, t_max, scene );
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

float IntersectRaySphere ( Vect3D O, Vect3D D, Sphere sphere, float dDot )
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

Intersection ClosestIntersection ( Vect3D O,
								   Vect3D D,
								   float t_min,
								   float t_max,
								   Sphere *scene )
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

uint16_t ClampColor ( uint16_t color )
{
	return max( 0, min( 255, color ) );
}

COLORREF CalculateFinalColor ( uint16_t &r,
							   uint16_t &g,
							   uint16_t &b,
							   COLORREF &reflected_color,
							   float &reflectiveness )
{
	uint16_t reflected_r =
		static_cast<uint16_t>( GetRValue( reflected_color ) * reflectiveness );
	uint16_t reflected_g =
		static_cast<uint16_t>( GetGValue( reflected_color ) * reflectiveness );
	uint16_t reflected_b =
		static_cast<uint16_t>( GetBValue( reflected_color ) * reflectiveness );

	return RGB( ClampColor( r * ( 1 - reflectiveness ) + ( reflected_r ) ),
				ClampColor( g * ( 1 - reflectiveness ) + ( reflected_g ) ),
				ClampColor( b * ( 1 - reflectiveness ) + ( reflected_b ) ) );
}

COLORREF TraceRay ( Vect3D origin,
					Vect3D destination,
					float t_min,
					float t_max,
					int recursion_depth,
					Sphere *scene,
					Light *lights )
{
	Vect3D intersection_sphere_normal = { };
	Vect3D origin_to_destination      = { };
	Vect3D reflected_ray              = { };

	Intersection intersection =
		ClosestIntersection( origin, destination, t_min, t_max, scene );

	Sphere *closest_sphere           = intersection.sphere;
	float closest_intersection_point = intersection.point;

	/* No intersecting sphere = empty space */
	if ( closest_sphere == NULL )
	{
		return RGB( 0, 0, 0 );
	}

	/* Compute intersection */
	origin_to_destination =
		origin + ( destination * closest_intersection_point );

	/* Computer sphere normal at intersection */
	intersection_sphere_normal =
		( origin_to_destination - closest_sphere->center ).norm( );

	/* Calculate light modification to color of the point */
	float color_lighting_modifier = CalcLight( origin_to_destination,
											   intersection_sphere_normal,
											   destination.invert( ),
											   closest_sphere->specularity,
											   scene,
											   lights );


	/* Split colors into R, G, and B and apply lightning modifier */
	uint16_t red   = static_cast<uint16_t>( GetRValue( closest_sphere->color )
                                          * color_lighting_modifier );
	uint16_t blue  = static_cast<uint16_t>( GetBValue( closest_sphere->color )
                                           * color_lighting_modifier );
	uint16_t green = static_cast<uint16_t>( GetGValue( closest_sphere->color )
											* color_lighting_modifier );


	float sphere_reflectivness = closest_sphere->reflective;

	/* Return if no more reflections need to be calculated */
	if ( recursion_depth <= 0 || sphere_reflectivness <= 0 )
	{
		return RGB( ClampColor( red ),
					ClampColor( green ),
					ClampColor( blue ) );
	}

	reflected_ray =
		ReflectRay( destination.invert( ), intersection_sphere_normal );

	/* Calculate color reflection */
	COLORREF reflected_color = TraceRay( origin_to_destination,
										 reflected_ray,
										 t_min,
										 t_max,
										 recursion_depth - 1,
										 scene,
										 lights );

	return CalculateFinalColor( red,
								green,
								blue,
								reflected_color,
								sphere_reflectivness );
}

void Draw ( BYTE **pLpvBits,
			int width,
			int height,
			Camera cam,
			Sphere *scene,
			Light *lights )
{
	Vect3D pojection_plane_point = { };
	int recursionDepth           = 1;
	float t_min                  = 0.001; // Epsilon do
	float t_max                  = INFINITY;

	for ( int x = 0; ( x < ( width ) ); ++x )
	{
		for ( int y = 0; ( y < ( height ) ); ++y )
		{
			pojection_plane_point = CanvasToViewport( x, y, width, height );
			pojection_plane_point =
				cam.ApplyCameraRotation( pojection_plane_point, cam );

			COLORREF color = TraceRay( cam.position,
									   pojection_plane_point,
									   t_min,
									   t_max,
									   recursionDepth,
									   scene,
									   lights );


			int offset = ( y * width + x ) * 4;
			if ( offset >= 0 && offset < width * height * 4 - 4 )
			{
				( *pLpvBits )[offset + 0] =
					static_cast<uint16_t>( GetBValue( color ) );
				( *pLpvBits )[offset + 1] =
					static_cast<uint16_t>( GetGValue( color ) );
				( *pLpvBits )[offset + 2] =
					static_cast<uint16_t>( GetRValue( color ) );
				( *pLpvBits )[offset + 3] = 255;
			}
		}
	}
}
