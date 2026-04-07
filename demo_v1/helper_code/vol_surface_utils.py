def slice_chain_by_volume(chain_dte, low_volume_threshold=3):
    """
    Slice options chain for a given DTE based on consecutive zero volumes.
    
    Starting from ATM strike:
    - Remove call wing if two consecutive low volume cVolu found above
    - Remove put wing if two consecutive low volume pVolu found below
    
    Parameters:
    -----------
    chain_dte : pd.DataFrame
        Options chain for a specific DTE, sorted by strike
    
    Returns:
    --------
    pd.DataFrame : Sliced chain with OTM options removed based on volume criteria
    """
    chain_dte = chain_dte.sort_values("strike").reset_index(drop=True).copy()
    
    if len(chain_dte) == 0:
        return chain_dte
    
    # Find ATM strike (closest to stock price)
    S = float(chain_dte["stkPx"].iloc[0])
    chain_dte["strike_diff"] = (chain_dte["strike"] - S).abs()
    atm_idx = chain_dte["strike_diff"].idxmin()
    
    # Initialize cut indices
    zeros = 0
    cut_put = -1   # Remove everything below (put wing)
    cut_call = len(chain_dte)  # Remove everything above (call wing)
    
    # Truncate put wing (walk downward from ATM, looking for 2 consecutive low volume pVolu)
    for i in range(atm_idx - 1, -1, -1): # from atm_idx-1 to 0 inclusive
        pVolu = chain_dte.iloc[i].get("pVolu", 0)
        if pVolu < low_volume_threshold:
            zeros += 1
        else:
            zeros = 0
        if zeros >= 2:
            cut_put = i  # Mark this index; everything <= i will be excluded
            break
    
    # Truncate call wing (walk upward from ATM, looking for 2 consecutive low volume cVolu)
    zeros = 0
    for i in range(atm_idx + 1, len(chain_dte)):
        cVolu = chain_dte.iloc[i].get("cVolu", 0)
        if cVolu < low_volume_threshold:
            zeros += 1
        else:
            zeros = 0
        if zeros >= 2:
            cut_call = i  # Mark this index; everything >= i will be excluded
            break
    
    # Apply slicing
    if cut_put >= 0:
        chain_dte = chain_dte.iloc[cut_put + 1:]
    if cut_call < len(chain_dte):
        chain_dte = chain_dte.iloc[:cut_call]
    
    return chain_dte.drop("strike_diff", axis=1)

