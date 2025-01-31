#pragma once
/*---------------------Includes---------------------------*/
#include <Windows.h>

/**
 * @brief Clamps color channel between 0 and 255
 *
 * @param color - Color channel
 *
 * @return Returns a uint16_t color channel thats been clamped
 */
static uint32_t ClampColor ( uint32_t  color )
{
    return max( 0, min( 255, color ) );
}

/**
 * @brief Applies multiplier to color, channel wise
 *
 * @param[in] color - Color, COLOREF
 * @param[in] multiplier - Multiplier
 *
 * @return Color product of color and multiplier
 */
COLORREF ApplyMultiplierToColor ( _In_ const COLORREF &color,
                                  _In_ const float &multiplier )
{
    uint32_t red   = static_cast<uint32_t>( GetRValue( color ) * multiplier );
    uint32_t blue  = static_cast<uint32_t>( GetBValue( color ) * multiplier );
    uint32_t green = static_cast<uint32_t>( GetGValue( color ) * multiplier );

    return RGB( red, blue, green );
}

/**
 * @brief Calculates final color of point, taking reflectiveness and reflection
 * modifiers into account
 *
 * @param r - Red after light calculations
 * @param g - Green after light caluclations
 * @param b - Blue after light calculations
 * @param reflected_color - Color from inverted ray of reflection
 * @param reflectiveness - Reflectiveness of the sphere where the point exists
 *
 * @return
 */
COLORREF CalculateFinalColor ( uint32_t &r,
                               uint32_t &g,
                               uint32_t &b,
                               COLORREF &lit_color,
                               COLORREF &reflected_color,
                               float &reflectiveness )
{
    float lit_color_multiplier = ( 1 - reflectiveness );

	// TODO: fix this error fu gavno
    uint32_t rr = static_cast<uint32_t>(GetRValue( lit_color ));
    uint32_t gg = static_cast<uint32_t>(GetGValue( lit_color ));
    uint32_t bb = static_cast<uint32_t>(GetBValue( lit_color ));


    uint32_t reflected_r =
        static_cast<uint32_t>( GetRValue( reflected_color ) * reflectiveness );
    uint32_t reflected_g =
        static_cast<uint32_t>( GetGValue( reflected_color ) * reflectiveness );
    uint32_t reflected_b =
        static_cast<uint32_t>( GetBValue( reflected_color ) * reflectiveness );

    return RGB(
        ClampColor( ClampColor( r * lit_color_multiplier ) + ( reflected_r ) ),
        ClampColor( ClampColor( g * lit_color_multiplier ) + ( reflected_g ) ),
        ClampColor( ClampColor( b * lit_color_multiplier )
                    + ( reflected_b ) ) );
}

/**
 * @brief Sets pixel at offset to color
 *
 * @param[in,out] p_lpv_bits - Pointer to pixel buffer
 * @param[in]     offset - Offset into pixel buffer for wanted pixel
 * @param[in]     color - Wanted color of pixel
 */
void SetPixelToColor ( _Inout_ BYTE **p_lpv_bits,
                       _In_ uint32_t &offset,
                       _In_ COLORREF &color )
{
    ( *p_lpv_bits )[offset + 0] = static_cast<uint16_t>( GetBValue( color ) );
    ( *p_lpv_bits )[offset + 1] = static_cast<uint16_t>( GetGValue( color ) );
    ( *p_lpv_bits )[offset + 2] = static_cast<uint16_t>( GetRValue( color ) );
    ( *p_lpv_bits )[offset + 3] = 255;
}

/*------------Template Declarations---------------*/
template<typename T>
concept Scalar = std::is_scalar_v<T>;

/**
 * @brief Returns true if a value is between, low and high
 *
 * @tparam T - Any scalar, std::is_scalar_v<T> == true
 *
 * @param value - value to check if its in bounds
 * @param low - lower bound
 * @param high - upper bound
 *
 * @return
 */
template<Scalar ValueType, Scalar LowType, Scalar HighType>
bool IsInBounds ( const ValueType &value,
                  const LowType &low,
                  const HighType &high )
{
    // Comparison logic will go here
    return !( value < low ) && ( value < high );
}
