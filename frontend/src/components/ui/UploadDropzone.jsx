import React, { useCallback, useState } from 'react';

/**
 * Dropzone minimaliste pour déposer une radio et la prévisualiser
 */
export function UploadDropzone({ onFileSelected, disabled }) {
  const [isDragging, setIsDragging] = useState(false);
  const [preview, setPreview] = useState(null);

  const handleFiles = useCallback(
    (files) => {
      const file = files?.[0];
      if (!file) return;
      setPreview(URL.createObjectURL(file));
      onFileSelected?.(file);
    },
    [onFileSelected]
  );

  const onDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    handleFiles(e.dataTransfer.files);
  };

  return (
    <div
      className={`relative border-2 border-dashed rounded-2xl p-6 transition-all duration-200 ${
        isDragging ? 'border-blue-500 bg-blue-500/5' : 'border-white/30 bg-white/5'
      } ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={disabled ? undefined : onDrop}
      onClick={() => {
        if (disabled) return;
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = 'image/*';
        input.onchange = (e) => handleFiles(e.target.files);
        input.click();
      }}
    >
      <div className="flex items-center gap-4">
        <div className="h-16 w-16 rounded-xl bg-blue-500/20 flex items-center justify-center border border-blue-400/40">
          <svg className="w-10 h-10 text-blue-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 15a4 4 0 004 4h10a4 4 0 004-4M7 10l5-5m0 0l5 5m-5-5v12" />
          </svg>
        </div>
        <div>
          <p className="text-white font-semibold">Glissez-déposez une radiographie</p>
          <p className="text-sm text-gray-300">PNG, JPG. Taille max dépend du serveur.</p>
        </div>
      </div>

      {preview && (
        <div className="mt-4 overflow-hidden rounded-xl border border-white/10 bg-black/30">
          <img src={preview} alt="Prévisualisation" className="w-full object-contain max-h-80" />
        </div>
      )}
    </div>
  );
}

